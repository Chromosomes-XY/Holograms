"""Explore rotated solid rectangular beams under a true 75% frontal opening."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,hashlib,time
from pathlib import Path
import numpy as np
from .benchmark_mesh import PreparedMesh
from .optimise_measured_noise import FC,POWER,WEIGHT,EDGES

OUT=Path(__file__).parent/'results/angled_rectangles'

def design(length,thickness,rows=1,gap=20,alternating=False):
    assert all(v%5==0 for v in [length,thickness,gap]) and thickness<=75<=length
    theta=float(np.arcsin(75/np.hypot(length,thickness))-np.arctan2(thickness,length))
    if abs(theta)<1e-12:theta=0.
    projected=length*np.sin(theta)+thickness*np.cos(theta)
    span=length*np.cos(theta)+thickness*np.sin(theta)
    first=5*np.ceil((10+span/2)/5);step=5*np.ceil((span+gap)/5)
    centres=[float(first+i*step) for i in range(rows)];depth=float(max(350,5*np.ceil((centres[-1]+span/2+10)/5)))
    assert depth<=600 and abs(projected-75)<1e-9
    return dict(length_mm=length,thickness_mm=thickness,rows=rows,min_axial_gap_mm=gap,alternating=alternating,
                angle_deg=float(np.rad2deg(theta)),centres_mm=centres,depth_mm=depth,projected_blockage_mm=projected,clear_fraction=.75)

def geometry(d,h=.0025):
    nx=int(np.ceil(.3/h));nz=int(np.ceil((d['depth_mm']+40)/1000/h))
    x=np.linspace(0,.3,nx+1);z=np.linspace(-.02,(d['depth_mm']+20)/1000,nz+1)
    X=(x[:-1]+x[1:])[None,:]/2-.15;Z=(z[:-1]+z[1:])[:,None]/2;solid=np.zeros((nz,nx),bool)
    if 'beams' in d:
        for beam in d['beams']:
            a=np.deg2rad(beam['angle_deg']);dx=X-(beam['x_mm']-150)/1000;dz=Z-beam['z_mm']/1000
            for shift in [-.3,0,.3]:
                along=dz*np.cos(a)+(dx+shift)*np.sin(a);across=(dx+shift)*np.cos(a)-dz*np.sin(a)
                solid |= (abs(along)<beam['length_mm']/2000-1e-12)&(abs(across)<beam['thickness_mm']/2000-1e-12)
        numerical_clear=float(np.diff(x)[~solid.any(axis=0)].sum()/.3)
        assert numerical_clear>=d.get('minimum_clear_fraction',.75)-2*h/.3-1e-10
        return solid,x,z,numerical_clear
    theta=np.deg2rad(d['angle_deg']);L=d['length_mm']/1000;t=d['thickness_mm']/1000
    for i,centre in enumerate(d['centres_mm']):
        a=-theta if d['alternating'] and i%2 else theta;dz=Z-centre/1000
        along=dz*np.cos(a)+X*np.sin(a);across=X*np.cos(a)-dz*np.sin(a)
        solid |= (abs(along)<L/2-1e-12)&(abs(across)<t/2-1e-12)
    numerical_clear=float(np.diff(x)[~solid.any(axis=0)].sum()/.3)
    assert numerical_clear>=.75-1e-10
    return solid,x,z,numerical_clear

def evaluate(d,n=3,h=.0025,dominant_n=None,angle=0):
    key=hashlib.sha256(json.dumps([d,n,h,dominant_n,angle,'rotated_v1'],sort_keys=True).encode()).hexdigest()[:20];path=OUT/'cache'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    s,x,z,clear=geometry(d,h);mesh=PreparedMesh(s,x,z,boundary='periodic');bands=[];J=0.
    for i,f in enumerate(FC):
        if POWER[i]==0:bands.append(dict(f=float(f),T=None,after=0));continue
        order=dominant_n if dominant_n and 630<=f<=1600 else n
        nodes,w=np.polynomial.legendre.leggauss(order);lo,hi=EDGES[i:i+2];fs=(lo+hi)/2+(hi-lo)*nodes/2;ts=[];errors=[]
        for frequency in fs:
            if angle:
                from .bloch_check import solve_bloch
                q=solve_bloch(s,x,z,float(frequency),angle);T=q['T'];err=q['energy_error']
            else:
                q=mesh.solve(float(frequency));T=q.transmission;err=q.balance_error
            assert np.isfinite(T) and -1e-7<=T<=1+1e-7 and err<1e-7
            ts.append(T);errors.append(err)
        avg=float(np.array(ts)@w/2);J+=WEIGHT[i]*avg
        bands.append(dict(f=float(f),T=avg,after=float(POWER[i]*avg),frequencies=fs.tolist(),transmission=ts,max_energy_error=max(errors)))
    r=dict(design=d,J=float(J),reduction_db=float(-10*np.log10(J)),n=n,dominant_n=dominant_n,h=h,incidence_deg=angle,numerical_clear_fraction=clear,bands=bands)
    path.write_text(json.dumps(r));return r

def candidates():
    ds=[design(430,75)]
    for length in [200,300,400,500]:ds.append(design(length,15))
    for length in [150,200]:
        for gap in [20,50]:
            for alternating in [False,True]:ds.append(design(length,15,2,gap,alternating))
    for length in [100,150]:
        for gap in [20,50]:ds.append(design(length,15,3,gap,True))
    for gap in [20,50]:
        for alternating in [False,True]:ds.append(design(100,15,4,gap,alternating))
    ds.extend([design(100,75,3,20),design(100,75,4,20)])
    return ds

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True);history=[]
    for i,d in enumerate(candidates()):
        r=evaluate(d);history.append(r);(OUT/'screening.json').write_text(json.dumps(history))
        print('Screen',i+1,'rows',d['rows'],'angle',round(d['angle_deg'],1),'depth',d['depth_mm'],'J',round(r['J'],5),flush=True)
    # Compare screened leaders and the initial straight 430 mm control using denser frequency sampling.
    finalists=[];selected_designs=[]
    for q in sorted(history,key=lambda q:q['J'])[:3]+[history[0]]:
        if q['design'] not in selected_designs:selected_designs.append(q['design'])
    for d in selected_designs:
        r=evaluate(d,n=9,h=.0025,dominant_n=33);finalists.append(r)
        print('Finalist',d['rows'],round(d['angle_deg'],1),r['J'],flush=True)
    best=min(finalists,key=lambda q:q['J'])
    report=dict(status='Bounded geometry survey, not a global optimisation',candidates=len(history),finalists=finalists,best=best,
        projection_rule='L sin(abs(theta)) + t cos(theta) = 75 mm; every row has the same projected interval in a 300 mm period',
        manufacturing='Lengths, thicknesses and beam centre locations use 5 mm increments. Angles are derived continuously to retain exactly 75% ideal clear frontal view. Derived corner coordinates are not stock-cut dimensions.',
        limitations=['Normal-incidence periodic rigid lossless 2D model, not a finite 9.9 m wall','Stair-step approximation of inclined boundaries; numerical opening can be slightly larger than the exact geometric opening','Three points per band for screening; dominant bands refined to 33 for finalists','High-frequency weak bands have limited spatial resolution','No grounds, supports, vertical-height effects, damping or structural design'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report,history)

def plot(r,history):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig=plt.figure(figsize=(14,11),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1.3,1,1]);a=fig.add_subplot(gs[0,:2]);b=fig.add_subplot(gs[0,2])
    for q in r['finalists']:
        d=q['design'];label=f'{d["rows"]} row(s), {d["angle_deg"]:.1f}°, {d["depth_mm"]:.0f} mm'
        a.semilogx(FC,[row['T'] if row['T'] is not None else np.nan for row in q['bands']],'o-',ms=3,label=label)
    a.set(xlim=(250,2500),ylim=(0,1.05),xlabel='Band centre (Hz)',ylabel='Band-averaged power transmission',title='Frequency-refined finalists');a.legend(fontsize=8);a.grid(alpha=.2)
    for rows in [1,2,3,4]:
        qs=[q for q in history if q['design']['rows']==rows];b.scatter([q['design']['depth_mm'] for q in qs],[q['reduction_db'] for q in qs],label=f'{rows} row(s)')
    b.set(xlabel='Overall depth (mm)',ylabel='Screening reduction (dB)',title='Coarse survey');b.legend(fontsize=8);b.grid(alpha=.2)
    ds=[q['design'] for q in sorted(r['finalists'],key=lambda q:q['J'])]
    for d in candidates():
        if len(ds)>=6:break
        if d not in ds and d['rows'] in [3,4]:ds.append(d)
    for ax,d in zip([fig.add_subplot(gs[row,col]) for row in [1,2] for col in range(3)],ds):
        for i,zc in enumerate(d['centres_mm']):
            angle=np.deg2rad(d['angle_deg'])*(-1 if d['alternating'] and i%2 else 1)
            u=np.array([-1,1,1,-1])*d['length_mm']/2;v=np.array([-1,-1,1,1])*d['thickness_mm']/2
            z=zc+u*np.cos(angle)-v*np.sin(angle);x=150+u*np.sin(angle)+v*np.cos(angle)
            ax.add_patch(Polygon(np.c_[z,x],color='#354c5c'))
        ax.axhspan(112.5,187.5,color='#cf9b4530');ax.set(xlim=(0,600),ylim=(0,300),xlabel='Depth (mm)',ylabel='Across unit (mm)',title=f'{d["rows"]} row(s) · {d["angle_deg"]:.1f}° · '+('alternating' if d['alternating'] else 'parallel'));ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Angled solid rectangular beams · 75% clear frontal view retained',fontsize=18)
    fig.supxlabel('Plan-view cross-sections. All row shadows overlap inside the same 75 mm strip; 225 mm stays clear.\nRigid, lossless periodic-unit screening. Angled boundaries are approximated by a Cartesian mesh.',fontsize=11)
    fig.savefig(OUT/'survey.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
