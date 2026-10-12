"""Multi-start discrete coordinate descent; 100 mm positions, 5 mm stock sizes."""
import json
from pathlib import Path
from .angled_rectangles import geometry,evaluate,OUT as CACHEOUT
from .benchmark_mesh import PreparedMesh
from .optimise_measured_noise import FC,POWER,WEIGHT,EDGES
import numpy as np
OUT=Path(__file__).parent/'results/discrete_beam_descent'
# L1,t1,angle1,L2,t2,angle2,rear lateral index,rear depth index.
STEPS=[25,5,5,25,5,5,1,1]
BOUNDS=[(75,300),(20,60),(-45,45),(75,300),(20,60),(-60,60),(-1,1),(3,6)]

def design(v):
    if any(not lo<=a<=hi for a,(lo,hi) in zip(v,BOUNDS)):raise ValueError('bounds')
    beams=[]
    for i in range(2):
        L,t,a=v[3*i:3*i+3];rad=np.deg2rad(a)
        B=L*abs(np.sin(rad))+t*abs(np.cos(rad));span=L*abs(np.cos(rad))+t*abs(np.sin(rad))
        beams.append(dict(length_mm=L,thickness_mm=t,angle_deg=a,projected_width_mm=float(B),axial_span_mm=float(span),x_mm=150 if i==0 else 150+100*v[6],z_mm=200 if i==0 else 100*v[7]))
    if beams[0]['projected_width_mm']>75+1e-9:raise ValueError('first-layer opening')
    if beams[1]['z_mm']-beams[0]['z_mm']<(beams[0]['axial_span_mm']+beams[1]['axial_span_mm'])/2+10:raise ValueError('layer separation')
    intervals=[]
    for b in beams:
        for shift in [-300,0,300]:
            lo=max(0,b['x_mm']+shift-b['projected_width_mm']/2);hi=min(300,b['x_mm']+shift+b['projected_width_mm']/2)
            if hi>lo:intervals.append((lo,hi))
    end=0;blocked=0
    for lo,hi in sorted(intervals):
        blocked+=max(0,hi-max(lo,end));end=max(end,hi)
    clear=1-blocked/300
    if clear<.5-1e-10:raise ValueError('combined opening')
    depth=float(5*np.ceil((beams[1]['z_mm']+beams[1]['axial_span_mm']/2+10)/5))
    return dict(beams=beams,rows=2,depth_mm=depth,clear_fraction=float(clear),first_layer_clear_fraction=1-beams[0]['projected_width_mm']/300,minimum_clear_fraction=.5,parameters=list(v),position_step_mm=100)

def screen(d):
    s,x,z,clear=geometry(d,.005);mesh=PreparedMesh(s,x,z,boundary='periodic');J=0
    for i,f in enumerate(FC):
        if f>3150:J+=WEIGHT[i];continue
        nodes,w=np.polynomial.legendre.leggauss(3);lo,hi=EDGES[i:i+2]
        for freq,weight in zip((lo+hi)/2+(hi-lo)*nodes/2,w/2):
            q=mesh.solve(float(freq));assert q.balance_error<1e-7 and -1e-7<=q.transmission<=1+1e-7
            J+=WEIGHT[i]*weight*q.transmission
    return dict(design=d,J=float(J))

def main():
    OUT.mkdir(parents=True,exist_ok=True);(CACHEOUT/'cache').mkdir(parents=True,exist_ok=True)
    path=OUT/'history.json';history=json.loads(path.read_text()) if path.exists() else []
    cache={tuple(r['design']['parameters']):r for r in history};traces=[];leaders=[]
    def get(v):
        k=tuple(v)
        if k not in cache:
            d=design(v);r=screen(d);cache[k]=r;history.append(r);path.write_text(json.dumps(history));print('Evaluation',len(history),'J',round(r['J'],5),'v',v,flush=True)
        return cache[k]
    seeds=[[150,25,15,150,25,-25,1,4],[100,25,-25,200,25,15,-1,5],[200,25,10,200,25,-25,0,5]]
    for seed in seeds:
        current=get(seed);trace=[current['J']]
        for sweep in range(3):
            old=current['J']
            for j,step in enumerate(STEPS):
                candidates=[current]
                for sign in [-1,1]:
                    v=current['design']['parameters'].copy();v[j]+=sign*step
                    try:candidates.append(get(v))
                    except ValueError:pass
                current=min(candidates,key=lambda r:r['J']);trace.append(current['J'])
            if old-current['J']<1e-5:break
        # Finish stock-length directions at the manufacturing resolution.
        for j in [0,3]:
            candidates=[current]
            for sign in [-1,1]:
                v=current['design']['parameters'].copy();v[j]+=sign*5
                try:candidates.append(get(v))
                except ValueError:pass
            current=min(candidates,key=lambda r:r['J']);trace.append(current['J'])
        leaders.append(current);traces.append(trace)
    refined=[]
    for r in leaders:
        q=evaluate(r['design'],n=5,h=.0025,dominant_n=17);refined.append(q);print('Refined',q['J'],flush=True)
    best=min(refined,key=lambda r:r['J']);best=evaluate(best['design'],n=9,h=.0025,dominant_n=33)
    initial=evaluate(design(seeds[0]),n=9,h=.0025,dominant_n=33)
    # A finer grid at identical quadrature isolates the spatial discretisation change.
    meshcheck=evaluate(best['design'],n=3,h=.00125,dominant_n=9)
    meshreference=evaluate(best['design'],n=3,h=.0025,dominant_n=9)
    report=dict(best=best,initial=initial,finalists=refined,evaluations=len(history),traces=traces,meshcheck_J=meshcheck['J'],meshreference_J=meshreference['J'],method='Three-start bounded discrete coordinate descent; not a differentiable gradient or a global optimum.',screen='5 mm mesh; 3 quadrature points per band through 3150 Hz; higher bands conservatively unchanged.',constraints='100 mm beam-centre placement increments; first centre fixed at (150,200) mm as reference; stock sizes multiples of 5 mm; angles searched in 5 degree steps; first layer >=75% and combined >=50% geometrical frontal opening.')
    beams=best['design']['beams']
    report['physical_beam_envelope_depth_mm']=max(b['z_mm']+b['axial_span_mm']/2 for b in beams)-min(b['z_mm']-b['axial_span_mm']/2 for b in beams)
    (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report)

def plot(r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,q,title in [(axs[0,0],r['initial'],'Starting layout'),(axs[0,1],r['best'],'Best refined layout')]:
        d=q['design']
        for shift in [-300,0,300]:
            for b in d['beams']:
                a=np.deg2rad(b['angle_deg']);u=np.array([-1,1,1,-1])*b['length_mm']/2;v=np.array([-1,-1,1,1])*b['thickness_mm']/2
                ax.add_patch(Polygon(np.c_[b['z_mm']+u*np.cos(a)-v*np.sin(a),b['x_mm']+shift+u*np.sin(a)+v*np.cos(a)],color='#31566b'))
        for i,b in enumerate(d['beams']):
            ax.annotate(f"{b['length_mm']} × {b['thickness_mm']} mm\n{b['angle_deg']:+}°",(b['z_mm'],b['x_mm']),xytext=(b['z_mm'],245 if i==0 else 20),ha='center',fontsize=8,arrowprops=dict(arrowstyle='-',color='#777777'))
        ax.set(xlim=(0,800),ylim=(0,300),xlabel='Depth (mm); sound travels →',ylabel='Across wall (mm)',title=f'{title}: {100*q["J"]:.1f}% weighted power transmitted\nFirst layer {100*d["first_layer_clear_fraction"]:.1f}% clear; combined {100*d["clear_fraction"]:.1f}% clear');ax.set_aspect('equal');ax.grid(alpha=.2)
    ax=axs[1,0]
    for q,label in [(r['initial'],'Starting layout'),(r['best'],'Optimised layout')]:ax.semilogx(FC,[b['T'] if b['T'] is not None else np.nan for b in q['bands']],'o-',label=label,ms=3)
    ax.set(xlim=(100,4000),ylim=(0,1.05),xlabel='Band centre (Hz)',ylabel='Band-averaged power transmission');ax.legend();ax.grid(alpha=.2)
    ax=axs[1,1]
    for i,trace in enumerate(r['traces']):ax.plot(np.arange(len(trace)),100*np.array(trace),label=f'Start {i+1}')
    ax.set(xlabel='Accepted/current coordinate state',ylabel='Screening weighted transmission (%)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Two solid-beam layers • discrete placement optimisation',fontsize=17)
    fig.supxlabel('Periodic 2D, rigid lossless beams, normal incidence. Inclined surfaces use a stair-step grid.\n50% combined clear view is an exploratory constraint; supports and finite wall height are excluded.',fontsize=10)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
