"""Finite-array broadband comparison for the selected 5 mm six-cavity slat."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,time,csv
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .finite_array_2d import FiniteArray
from .manufacturable_search import OUT as DESIGN,geometry,layout,FC,P,G
from .benchmark_mesh import segmented_grid

OUT=Path(__file__).parent/'results/finite_six_5mm_comparison'
SELECTED=next(q for q in json.loads((DESIGN/'report.json').read_text())['selected'] if q['cavity_count']==6)
V=SELECTED['v'];REFINED=[630,800,1000,1250,1600]

class SixArray(FiniteArray):
    def __init__(self,h=.02,far=.04,pml=.75,filled=False):
        self.h=h;self.far=far;self.pml=pml;self.cells=33
        unit,ux,uz=geometry(V,h);cs,depth_mm=layout(V);self.depth=depth=depth_mm/1000;half=4.95
        self.xe=np.unique(np.round(np.r_[segmented_grid([-6.75-pml,-6.75,-half],far),np.concatenate([ux+i*.3-half for i in range(33)]),segmented_grid([half,6.75,6.75+pml],far)],12))
        self.ze=np.unique(np.round(np.r_[segmented_grid([-3.75-pml,-3.75,-.02],far),uz,segmented_grid([depth+.02,4.75,4.75+pml],far)],12))
        self.wx=np.diff(self.xe);self.wz=np.diff(self.ze);self.x=(self.xe[:-1]+self.xe[1:])/2;self.z=(self.ze[:-1]+self.ze[1:])/2
        self.X,self.Z=np.meshgrid(self.x,self.z);local=(self.X+half)%.3;inside=abs(self.X)<half
        solid=inside&(local<.075)&(self.Z>0)&(self.Z<depth)
        if not filled:
            for c in cs:
                a=c['start']/1000;b=a+c['length']/1000;w=c['width']/1000;m=c['mouth']/1000
                rects=[(.005,.005+w,a,b),(0,.005,a,a+m)] if c['side']==0 else [(.07-w,.07,a,b),(.07,.075,a,a+m)]
                for l,r,f,t in rects:solid &= ~(inside&(local>l)&(local<r)&(self.Z>f)&(self.Z<t))
        self.solid=solid;self.ids=np.full(solid.shape,-1,int);self.ids[~solid]=np.arange((~solid).sum());self.n=int((~solid).sum())
        self.clear=float(self.wx[inside[0]&~solid.any(axis=0)].sum()/9.9);assert abs(self.clear-.75)<1e-10

def meshes():
    return {(fine,filled):SixArray(h=.01 if fine else .02,far=.02 if fine else .04,filled=filled) for fine in [False,True] for filled in [False,True]}

def main():
    OUT.mkdir(exist_ok=True);ms=meshes()
    for fine in [False,True]:
        a,b=ms[fine,False],ms[fine,True]
        assert np.array_equal(a.xe,b.xe) and np.array_equal(a.ze,b.ze) and np.all(~a.solid|b.solid)
        print('Mesh',fine,a.solid.shape,a.n,flush=True)
    manifest=dict(v=V,version=1,source=[0,-3],clear_fraction=.75,depth_m=.43,width_m=9.9)
    mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Model changed: use a new output directory'
    mp.write_text(json.dumps(manifest,indent=2))
    for f in FC[FC<=1600]:
        fine=f in REFINED
        for kind in ['no_wall','filled','cavities']:
            path=OUT/f'{kind}_{f:g}.npz'
            if path.exists():continue
            mesh=ms[fine,kind=='filled'];start=time.time();p,_,res=mesh.solve(float(f),empty=kind=='no_wall')
            np.savez_compressed(path,p=p,residual=res)
            print(kind,f,'Hz',round(time.time()-start,1),'s',flush=True)
    assemble(ms)

def assemble(ms):
    coarse=ms[False,False];coords=np.stack([coarse.Z.ravel(),coarse.X.ravel()],axis=-1)
    powers={kind:np.zeros(coarse.solid.shape) for kind in ['no_wall','filled','cavities']};tail=np.zeros(coarse.solid.shape);bands=[]
    for f,p,g in zip(FC,P,G):
        if not p:continue
        if f>1600:tail+=p*g*abs(coarse.incident(f))**2;continue
        fine=f in REFINED;mesh=ms[fine,False];area=mesh.wz[:,None]*mesh.wx[None,:];region=(abs(mesh.X)<4)&(mesh.Z>.7)&(mesh.Z<3.5);row=dict(frequency=float(f))
        for kind in powers:
            a=abs(np.load(OUT/f'{kind}_{f:g}.npz')['p'])**2;row[kind]=float(np.sum(a[region]*area[region]))
            if fine:a=RegularGridInterpolator((mesh.z,mesh.x),a,method='nearest')(coords).reshape(coarse.solid.shape)
            powers[kind]+=p*g*a
        row['cavity_advantage_db']=float(10*np.log10(row['filled']/row['cavities']));bands.append(row)
    for kind in powers:powers[kind]+=tail
    area=coarse.wz[:,None]*coarse.wx[None,:];region=(abs(coarse.X)<4)&(coarse.Z>.7)&(coarse.Z<3.5)
    totals={k:float(np.sum(a[region]*area[region])) for k,a in powers.items()}
    report=dict(geometry=SELECTED,source=[0,-3],cells=33,width_m=9.9,depth_m=coarse.depth,clear_fraction=coarse.clear,intended_height_m=1.8,
        filled_insertion_loss_db=float(10*np.log10(totals['no_wall']/totals['filled'])),cavity_insertion_loss_db=float(10*np.log10(totals['no_wall']/totals['cavities'])),cavity_advantage_db=float(10*np.log10(totals['filled']/totals['cavities'])),
        receiver_region='x=-4..4 m, z=0.7..3.5 m; integrate mean-square pressure before taking ratio, not transmitted flux',bands=bands,
        mesh_unknowns={str(fine):ms[fine,False].n for fine in [False,True]},
        weighted_fraction_solved=float((P*G)[FC<=1600].sum()/(P*G).sum()),
        limitations=['Horizontal 2D point source (3D line source), no over-top diffraction or ground',
                     'Band-centre solves through 1600 Hz; remaining 5.1% of weighted input added unchanged to both designs',
                     'Dominant 630–1600 Hz bands on refined mesh; no claim of spatial or within-band convergence',
                     'Rigid lossless walls and ideal sealed covers; no joints or structural vibration'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'broadband.npz',**powers,tail=tail,x=coarse.x,z=coarse.z)
    with (OUT/'regional_band_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(bands[0]));w.writeheader();w.writerows(bands)
    plot(ms,powers,report)
    print('RESULT',json.dumps({k:v for k,v in report.items() if k.endswith('_db')}),flush=True)

def plot(ms,powers,r):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    coarse=ms[False,False];fine=ms[True,False];norm=float((P*G).sum())
    fp=np.load(OUT/'filled_1000.npz')['p'];cp=np.load(OUT/'cavities_1000.npz')['p']
    fields=[10*np.log10(powers['filled']/norm),10*np.log10(powers['cavities']/norm),10*np.log10(powers['filled']/powers['cavities']),fp.real,cp.real,10*np.log10(abs(fp)**2/abs(cp)**2)]
    titles=['Solid-filled slats · weighted sound level','Six covered cavities · weighted sound level','Broadband advantage of cavities','Solid-filled slats · 1 kHz pressure','Six covered cavities · 1 kHz pressure','1 kHz advantage of cavities']
    fig,axs=plt.subplots(2,3,figsize=(19,11),layout='constrained')
    for i,(ax,a,title) in enumerate(zip(axs.flat,fields,titles)):
        mesh=ms[i>=3,i%3!=1];kw=dict(cmap='magma',vmin=-15,vmax=3) if i<2 else (dict(cmap='RdBu_r',vmin=-.8,vmax=.8) if i in [3,4] else dict(cmap='RdBu',norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12)))
        im=ax.pcolormesh(mesh.xe,mesh.ze,np.ma.masked_where(mesh.solid,a),shading='flat',rasterized=True,**kw)
        ax.contourf(mesh.x,mesh.z,mesh.solid,levels=[.5,1.5],colors=['#30363d']);ax.plot(0,-3,'*',color='#ffc400',ms=10,markeredgecolor='black')
        ax.set(xlim=(-6.5,6.5),ylim=(-3.6,4.5),xlabel='Across wall x (m)',ylabel='Propagation z (m)',title=title);ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,label='Relative weighted level (dB)' if i<2 else ('Relative pressure' if i in [3,4] else 'dB · positive = cavities quieter'))
    fig.suptitle('Best six-covered-cavity design versus solid-filled slats · 430 mm depth · 75% open',fontsize=18)
    fig.supxlabel(f'Receiver-region weighted reduction: filled {r["filled_insertion_loss_db"]:.2f} dB; cavities {r["cavity_insertion_loss_db"]:.2f} dB; cavity advantage {r["cavity_advantage_db"]:.2f} dB.\n9.9 m array, source 3 m in front. Horizontal 2D; no height/ground. Band centres through 1,600 Hz; higher-frequency tail added unchanged.',fontsize=12)
    fig.savefig(OUT/'comparison_maps.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    fs=[b['frequency'] for b in r['bands']]
    for kind,label,col in [('filled','Solid-filled','#8a8d91'),('cavities','Six covered cavities','#168777')]:
        axs[0].semilogx(fs,[10*np.log10(b['no_wall']/b[kind]) for b in r['bands']],'o-',label=label,color=col,ms=3)
        iz=int(np.argmin(abs(coarse.z-(coarse.depth+2))));axs[1].plot(coarse.x,10*np.log10(powers['no_wall'][iz]/powers[kind][iz]),label=label,color=col)
    axs[0].set(xlabel='Band centre (Hz)',ylabel='Receiver-region pressure reduction (dB)',xlim=(100,1600),title='Frequency comparison · no within-band averaging')
    axs[1].set(xlabel='Across wall x (m)',ylabel='Weighted pressure reduction (dB)',xlim=(-6,6),title='Receiver line approximately 2 m behind rear face')
    for ax in axs:ax.axhline(0,color='k',lw=.6);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'comparison_curves.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,3.5),layout='constrained')
    for ax,filled,title in zip(axs,[True,False],['Solid-filled rectangles','Six covered cavities per slat']):
        mesh=ms[True,filled];ax.pcolormesh(mesh.xe,mesh.ze,mesh.solid,cmap='Greys',shading='flat')
        ax.set(xlim=(-.6,.6),ylim=(-.03,.46),xlabel='Across wall x (m)',ylabel='Depth z (m)',title=title);ax.set_aspect('equal')
    fig.suptitle('Matched geometry: 430 mm depth, 75 mm slats, 225 mm clear gaps')
    fig.savefig(OUT/'geometry_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
