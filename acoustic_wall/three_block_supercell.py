"""Coupled ABC supercell of three manufacturable six-covered-cavity blocks."""
from .manufacturable_search import layout,geometry as single_geometry,evaluate as single_evaluate
from .angled_rectangles import PreparedMesh,FC,POWER,WEIGHT,EDGES
import numpy as np
import json,hashlib
from pathlib import Path
OUT=Path(__file__).parent/'results/three_block_supercell'
BASE=[80,30,35,95,20,50,85,30,40,120,15,40,115,25,30,65,25,30,45,65,60,45,35,20]

def blocks(p):
    vs=[]
    for j in range(3):
        v=BASE.copy();scale=p[j]/100;mouthshift=p[3+j]
        for i in range(6):
            v[3*i]=int(5*round(BASE[3*i]*scale/5))
            v[3*i+2]=int(np.clip(BASE[3*i+2]+mouthshift,15,min(50,v[3*i]-10)))
        for i in range(20,24):v[i]=int(5*round(BASE[i]*scale/5))
        layout(v);vs.append(v)
    return vs

def geometry(vs,h,depth=None):
    depth=depth or max(layout(v)[1] for v in vs)
    nx=round(.9/h);nz=round((depth+40)/1000/h)
    x=np.linspace(0,.9,nx+1);z=np.linspace(-.02,(depth+20)/1000,nz+1)
    X=(x[:-1]+x[1:])[None,:]*500;Z=(z[:-1]+z[1:])[:,None]*500
    solid=np.zeros((nz,nx),bool)
    for j,v in enumerate(vs):
        xx=X-300*j;part=(xx>0)&(xx<75)&(Z>0)&(Z<depth)
        for c in layout(v)[0]:
            a=c['start'];b=a+c['length'];w=c['width'];m=c['mouth']
            rects=[(5,5+w,a,b),(0,5,a,a+m)] if c['side']==0 else [(70-w,70,a,b),(70,75,a,a+m)]
            for l,r,lo,hi in rects:part &=~((xx>l)&(xx<r)&(Z>lo)&(Z<hi))
        solid|=part
    assert abs(np.diff(x)[~solid.any(axis=0)].sum()/.9-.75)<1e-9
    return solid,x,z,depth

def evaluate(p,h=.005,n=3,dominant_n=5,depth=None):
    key=hashlib.sha256(json.dumps([list(p),h,n,dominant_n,depth]).encode()).hexdigest()[:20];path=OUT/'cache'/f'{key}.json'
    if path.exists():return json.loads(path.read_text())
    vs=blocks(p);s,x,z,depth=geometry(vs,h,depth)
    # Exact smaller-period reduction for identical-block controls (validated below).
    if vs[0]==vs[1]==vs[2]:
        nx=s.shape[1]//3;s=s[:,:nx];x=x[:nx+1]
    mesh=PreparedMesh(s,x,z,boundary='periodic');bands=[];J=0
    for i,f in enumerate(FC):
        if f>3150:
            T=1.;err=0.;mode='unchanged upper tail'
        else:
            order=dominant_n if 630<=f<=1600 else n;nodes,w=np.polynomial.legendre.leggauss(order);lo,hi=EDGES[i:i+2];ts=[];errors=[]
            for freq in (lo+hi)/2+(hi-lo)*nodes/2:
                q=mesh.solve(float(freq));assert q.balance_error<1e-7 and -1e-7<=q.transmission<=1+1e-7
                ts.append(q.transmission);errors.append(q.balance_error)
            T=float(np.array(ts)@w/2);err=max(errors);mode='coupled solve'
        J+=WEIGHT[i]*T;bands.append(dict(f=float(f),T=T,after=float(POWER[i]*T),error=err,mode=mode))
    r=dict(parameters=list(p),blocks=vs,depth_mm=depth,J=float(J),reduction_db=float(-10*np.log10(J)),bands=bands,h=h,n=n,dominant_n=dominant_n,unchanged_tail_weight=float(WEIGHT[FC>3150].sum()))
    path.write_text(json.dumps(r));return r

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    # Repeated cells must reproduce the original 300 mm solver, including all modal power.
    s,x,z,_=geometry([BASE]*3,.005);big=PreparedMesh(s,x,z,boundary='periodic');s,x,z=single_geometry(BASE,.005);small=PreparedMesh(s,x,z,boundary='periodic')
    checks=[]
    for f in [800,1000,1250]:
        a=big.solve(f);b=small.solve(f);delta=abs(a.transmission-b.transmission);assert delta<1e-7;checks.append(dict(f=f,difference=delta))
    print('Repeated-cell equivalence passed',checks,flush=True)
    hist=[]
    def get(p):
        if any(v<80 or v>120 for v in p[:3]) or any(abs(v)>15 for v in p[3:]):raise ValueError()
        try:blocks(p)
        except AssertionError:raise ValueError()
        r=evaluate(p)
        if p not in [q['parameters'] for q in hist]:hist.append(r);(OUT/'history.json').write_text(json.dumps(hist));print('Candidate',len(hist),p,round(r['J'],6),flush=True)
        return r
    baseline=get([100,100,100,0,0,0]);current=get([80,100,120,0,0,0])
    for sweep in range(2):
        before=current['J']
        for i,step in enumerate([10,10,10,5,5,5]):
            candidates=[current]
            for sign in [-1,1]:
                p=current['parameters'].copy();p[i]+=sign*step
                try:candidates.append(get(p))
                except ValueError:pass
            current=min(candidates,key=lambda q:q['J'])
        if before-current['J']<1e-5:break
    leaders=sorted(hist,key=lambda q:q['J'])[:2];refined=[]
    for q in [baseline]+leaders:
        r=evaluate(q['parameters'],h=.0025,n=5,dominant_n=17);refined.append(r);print('Refined',r['parameters'],r['J'],flush=True)
    best=min(refined,key=lambda q:q['J']);dense=evaluate(best['parameters'],h=.0025,n=5,dominant_n=33)
    ref=evaluate(baseline['parameters'],h=.0025,n=5,dominant_n=33)
    matched=evaluate(baseline['parameters'],h=.0025,n=5,dominant_n=33,depth=dense['depth_mm']) if dense['depth_mm']!=430 else ref
    r=dict(best=dense,baseline=ref,matched_depth_baseline=matched,refined=refined,checks=checks,candidates=len(hist),method='Bounded coordinate descent in three block axial scales and three independent mouth offsets, snapped to 5 mm; not all 72 dimensions independently optimised.',assumptions='900 mm periodic supercell, 75% actual frontal opening, rigid lossless 2D normal incidence. All outgoing propagating modes included. Upper bands above 3150 Hz conservatively unchanged. Not a finite 9.9 m wall result.')
    (OUT/'report.json').write_text(json.dumps(r,indent=2));plot(r)

def plot(r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig,axs=plt.subplots(1,3,figsize=(16,7),layout='constrained')
    q=r['best'];s,x,z,_=geometry(q['blocks'],.0025,q['depth_mm'])
    axs[0].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#f4f8fa','#345a6b']),shading='flat');axs[0].set_aspect('equal')
    for j,label in enumerate('ABC'):axs[0].text(q['depth_mm']/2,300*j+160,label,fontsize=18,ha='center');axs[0].axhline(300*j,color='#aaa',lw=.6)
    axs[0].set(xlabel='Depth (mm); sound travels →',ylabel='Across wall (mm)',title='Three-block repeating unit; 75% clear')
    for result,label in [(r['baseline'],'Original A–A–A'),(r['matched_depth_baseline'],'A–A–A at matched depth'),(q,'Selected A–B–C')]:
        axs[1].semilogx(FC,[b['T'] for b in result['bands']],'o-',ms=3,label=f'{label}: {100*result["J"]:.1f}%')
        axs[2].semilogx(FC,[b['after'] for b in result['bands']],'o-',ms=3,label=label)
    axs[2].semilogx(FC,POWER,'--',color='gray',label='Before wall')
    for ax in axs[1:]:ax.set(xlim=(200,3150),xlabel='Band centre (Hz)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[1].set(ylabel='Band-averaged power transmission',ylim=(0,1.03));axs[2].set(ylabel='A-weighted band power')
    fig.suptitle('Coupled three-block cavity supercell • initial broadband search',fontsize=17)
    fig.supxlabel('Periodic 2D normal incidence; not the finite-wall receiver-region comparison. Dimensions on a 5 mm grid.\nAbove 3150 Hz, transmission is conservatively set to one in all cases.',fontsize=10)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
