"""Integer 5 mm search for four/six smooth cavities in a 75 mm solid slat."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,hashlib,time
from pathlib import Path
import numpy as np
from .benchmark_mesh import PreparedMesh,segmented_grid
from .optimise_measured_noise import FC,P,G,POWER,WEIGHT,EDGES

OUT=Path(__file__).parent/'results/manufacturable_5mm'

def layout(v):
    v=np.asarray(v,dtype=float);pairs=len(v)//8;n=2*pairs
    assert pairs in [2,3] and len(v)==8*pairs and np.allclose(v/5,np.round(v/5))
    starts=list(v[3*n:3*n+2]);cs=[]
    for j in range(pairs):
        for side in range(2):
            i=2*j+side;length,width,mouth=v[3*i:3*i+3]
            assert 50<=length<=150 and 15<=width<=30 and 15<=mouth<=min(50,length-10)
            a=starts[side]
            cs.append(dict(name=f'{"L" if side==0 else "R"}{j+1}',side=side,start=float(a),length=float(length),width=float(width),mouth=float(mouth)))
            if j<pairs-1:
                gap=v[3*n+2+2*j+side];assert 10<=gap<=110;starts[side]=a+length+gap
    assert np.all(v[3*n:3*n+2]>=10) and np.all(v[3*n:3*n+2]<=70)
    depth=max(350.,max(q['start']+q['length'] for q in cs)+10)
    assert depth<=600 and depth%5==0
    return cs,depth

def geometry(v,h=.005,notches=False):
    cs,depth=layout(v);xs=[0,75,300];zs=[-20,0,depth,depth+20];rects=[]
    for c in cs:
        a=c['start'];b=a+c['length'];w=c['width'];m=c['mouth']
        # Both faces have 5 mm mouth walls; widths <=30 leave >=5 mm central web.
        if notches:rs=[(0,5+w,a,b)] if c['side']==0 else [(70-w,75,a,b)]
        else:rs=[(5,5+w,a,b),(0,5,a,a+m)] if c['side']==0 else [(70-w,70,a,b),(70,75,a,a+m)]
        rects+=rs
    for a,b,c,d in rects:xs.extend([a,b]);zs.extend([c,d])
    assert np.allclose(np.array(xs)/5,np.round(np.array(xs)/5)) and np.allclose(np.array(zs)/5,np.round(np.array(zs)/5))
    x=segmented_grid(np.array(xs)/1000,h);z=segmented_grid(np.array(zs)/1000,h)
    X=(x[1:]+x[:-1])[None,:]*500;Z=(z[1:]+z[:-1])[:,None]*500
    solid=(X<75)&(Z>0)&(Z<depth)
    for a,b,c,d in rects:solid &= ~((X>a)&(X<b)&(Z>c)&(Z<d))
    assert abs(np.diff(x)[~solid.any(axis=0)].sum()/.3-.75)<1e-10
    return solid,x,z

def evaluate(v,n=5,h=.005,cutoff=None,angle=0,dominant_n=None,notches=False):
    v=list(map(int,v));key_data=[v,n,h,angle,dominant_n,'full_slat_v1']
    if notches:key_data.append('open_notches_v1')
    key=hashlib.sha256(json.dumps(key_data).encode()).hexdigest()[:20]
    path=OUT/'cache'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    rows=[];meshes={};total=0.
    for i in np.argsort(-WEIGHT):
        if POWER[i]==0:rows.append(dict(index=int(i),f=float(FC[i]),T=None,after=0));continue
        order=dominant_n if dominant_n and 630<=FC[i]<=1600 else n
        nodes,weights=np.polynomial.legendre.leggauss(order);lo,hi=EDGES[i:i+2];fs=(lo+hi)/2+(hi-lo)*nodes/2
        step=min(h,.0025) if hi>4000 else h
        if step not in meshes:
            s,x,z=geometry(v,step,notches=notches);meshes[step]=(PreparedMesh(s,x,z,boundary='periodic'),s,x,z)
        mesh,s,x,z=meshes[step];ts=[];errors=[]
        for f in fs:
            if angle:
                from .bloch_check import solve_bloch
                q=solve_bloch(s,x,z,float(f),angle);t=q['T'];error=q['energy_error']
            else:
                q=mesh.solve(float(f));t=q.transmission;error=q.balance_error
            assert np.isfinite(t) and -1e-7<=t<=1+1e-7 and error<1e-7
            ts.append(t);errors.append(error)
        t=float(np.array(ts)@weights/2);total+=WEIGHT[i]*t
        rows.append(dict(index=int(i),f=float(FC[i]),T=t,after=float(POWER[i]*t),frequencies=fs.tolist(),transmission=ts,max_energy_error=max(errors)))
        if cutoff is not None and total>=cutoff:return dict(v=v,pruned=True,lower_bound=float(total))
    rows.sort(key=lambda q:q['index']);cs,depth=layout(v)
    result=dict(v=v,pruned=False,J=float(total),reduction_db=float(-10*np.log10(total)),depth_mm=depth,cavities=cs,n=n,h=h,angle=angle,dominant_n=dominant_n,bands=rows,notches=notches)
    path.write_text(json.dumps(result));return result

SEEDS={4:[80,30,35,95,20,35,85,20,35,100,20,40,45,65,75,110],
       6:[80,30,35,95,20,35,85,20,35,100,20,40,115,25,40,65,25,30,45,65,35,45,20,20]}

def propose(v,rng,step):
    n=2*(len(v)//8)
    bounds=np.array([(50,150),(15,30),(15,50)]*n+[(10,70)]*2+[(10,110)]*(n-2))
    for attempt in range(1000):
        q=np.array(v);count=1 if step%3==0 else int(rng.integers(2,7));indices=rng.choice(len(v),count,replace=False)
        q[indices]+=5*rng.choice([-4,-3,-2,-1,1,2,3,4],size=count)
        q=np.clip(q,bounds[:,0],bounds[:,1]).astype(int)
        try:layout(q)
        except AssertionError:continue
        if q.tolist()!=list(v):return q.tolist()
    raise RuntimeError('No feasible discrete proposal')

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True);all_history={};final=[]
    for count,seed in SEEDS.items():
        hp=OUT/f'history_{count}.json'
        history=json.loads(hp.read_text()) if hp.exists() else []
        rng=np.random.default_rng(500+count)
        if not history:history=[evaluate(seed)];hp.write_text(json.dumps(history))
        best=min([q for q in history if not q['pruned']],key=lambda q:q['J'])
        # Saved proposals make continuation deterministic for a completed run.
        for step in range(len(history)-1,40):
            completed=sorted([q for q in history if not q['pruned']],key=lambda q:q['J'])
            parent=best if step%5 else completed[int(rng.integers(min(4,len(completed))))]
            v=propose(parent['v'],rng,step);q=evaluate(v,cutoff=best['J']);history.append(q)
            if not q['pruned'] and q['J']<best['J']:best=q;print('Improved',count,step+1,round(best['J'],5),'depth',best['depth_mm'],flush=True)
            hp.write_text(json.dumps(history))
            if (step+1)%10==0:print('Progress',count,step+1,'best',best['J'],flush=True)
        all_history[str(count)]=history
        completed=sorted([q for q in history if not q['pruned']],key=lambda q:q['J']);vs=[]
        for q in completed[:2]+[history[0]]:
            if q['v'] not in vs:vs.append(q['v'])
        for v in vs:
            q=evaluate(v,n=9,h=.0025);q['cavity_count']=count;q['seed']=v==seed;final.append(q)
            print('Finalist',count,q['J'],q['depth_mm'],flush=True)
    selected=[]
    for count in SEEDS:
        q=min([q for q in final if q['cavity_count']==count],key=lambda q:q['J'])
        checked=evaluate(q['v'],n=9,h=.0025,dominant_n=33);checked['cavity_count']=count
        checked['J_9point']=q['J'];selected.append(checked)
        print('Selected',count,checked['J'],checked['depth_mm'],flush=True)
    report=dict(status='Bounded discrete screening search, not global optimum',step_mm=5,pitch_mm=300,blocked_mm=75,clear_mm=225,clear_fraction=.75,
                depth_range_mm=[350,600],wall_thickness_mm=5,proposal_count_per_family=40,finalists=final,selected=selected,
                assumptions='Normal incidence, rigid lossless 2D periodic model. All nonzero supplied bands; uniform power per Hz within bands. No amplitude/phases prescribed independently of geometry.',
                limitations=['Limited local search; high openness and finite wall height remain constraints','No damping added in this first comparison','No finite-array or 1.8 m height validation for these new geometries','Numerical and quadrature convergence are finite; oblique checks are separate'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report,all_history)

def plot(report,hist):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axs=plt.subplots(3,2,figsize=(14,12),layout='constrained')
    colours=['#2582b8','#168b71']
    for q,col in zip(report['selected'],colours):
        label=f'{q["cavity_count"]} cavities · {q["depth_mm"]:.0f} mm · {q["reduction_db"]:.2f} dB'
        axs[0,0].semilogx(FC,[b['T'] if b['T'] is not None else np.nan for b in q['bands']],'o-',color=col,label=label,ms=3)
        axs[0,1].semilogx(FC,[b['after'] for b in q['bands']],'o-',color=col,label=label,ms=3)
    axs[0,1].semilogx(FC,POWER,color='#d18a31',label='Before wall')
    for ax in axs[0]:ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set(xlim=(100,5000),xlabel='Band centre (Hz)')
    axs[0,0].set(ylabel='Band-averaged power transmission',ylim=(0,1.05));axs[0,1].set(ylabel='Weighted band power')
    for ax,q in zip(axs[1],report['selected']):
        s,x,z=geometry(q['v'],.0025);ax.pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#edf4f5','#354450']),shading='flat');ax.set_aspect('equal');ax.set(xlabel='Depth (mm)',ylabel='Across unit (mm)',title=f'{q["cavity_count"]} cavities · all dimensions on 5 mm grid')
    for key,col in zip(['4','6'],colours):
        qs=[q for q in hist[key] if not q['pruned']];axs[2,0].scatter([q['depth_mm'] for q in qs],[q['reduction_db'] for q in qs],color=col,label=key+' cavities',alpha=.6)
    axs[2,0].set(xlabel='Depth (mm)',ylabel='Search estimate of weighted reduction (dB)',title='Completed search evaluations · coarse estimates');axs[2,0].legend();axs[2,0].grid(alpha=.2)
    axs[2,1].axis('off');axs[2,1].text(.02,.95,'Constraints\n\n300 mm pitch; 225 mm clear opening (75%).\n75 mm full slat; minimum 5 mm solid walls.\nDimensions varied in 5 mm increments.\nDepth explored from 350 to 600 mm.\n\nNormal incidence; all supplied bands included.\n33 samples in dominant bands for final comparison.\nLossless periodic 2D model; not an outdoor rating.',va='top',fontsize=12)
    fig.suptitle('Manufacturable broadband optimisation · four versus six cavities',fontsize=18);fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
