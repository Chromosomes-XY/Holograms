"""Finite 33-cell rigid wall: total-field FV Helmholtz with stretched PML.

Horizontal 2D section. Analytic Hankel incident wave is a 2D point/3D line source.
No homogenised screen or periodic boundary conditions are used in this solver.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,time,argparse
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import hankel1
from .expand_300 import geometry,layout
from .benchmark_mesh import segmented_grid
from .optimise_measured_noise import FC,P,G

OUT=Path(__file__).parent/'results/finite_wall_10m'
V=json.loads((Path(__file__).parent/'results/measured_noise_optimised/report.json').read_text())['selected']['v']

class FiniteArray:
    def __init__(self,h=.02,far=.04,pml=.75,cells=33,filled=False):
        self.h=h;self.far=far;self.pml=pml;self.cells=cells
        unit,ux,uz=geometry(V,h);depth=layout(V)[1]/1000;half=cells*.3/2
        # Preserve every cavity interface. Mesh is coarser away from the wall.
        self.xe=np.unique(np.round(np.r_[segmented_grid([-6.75-pml,-6.75,-half],far),
                          np.concatenate([ux+i*.3-half for i in range(cells)]),
                          segmented_grid([half,6.75,6.75+pml],far)],12))
        self.ze=np.unique(np.round(np.r_[segmented_grid([-3.75-pml,-3.75,-.02],far),uz,
                         segmented_grid([depth+.02,4.75,4.75+pml],far)],12))
        self.x=(self.xe[:-1]+self.xe[1:])/2;self.z=(self.ze[:-1]+self.ze[1:])/2
        self.wx=np.diff(self.xe);self.wz=np.diff(self.ze);self.depth=depth
        X,Z=np.meshgrid(self.x,self.z);self.X=X;self.Z=Z
        local=(X+half)%.3;inside=abs(X)<half
        solid=inside&((local<.0375)|(local>.2625))&(Z>0)&(Z<depth)
        cs,_=layout(V)
        if filled:cs=[]
        for q in cs:
            a=q['start']/1000;b=a+q['length']/1000;w=q['width']/1000;m=q['mouth']/1000
            rects=[(.0345-w,.0345,a,b),(.0345,.0375,a,a+m)] if q['side']=='left' else [(.2655,.2655+w,a,b),(.2625,.2655,a,a+m)]
            for l,r,f,t in rects:solid &= ~(inside&(local>l)&(local<r)&(Z>f)&(Z<t))
        self.solid=solid;self.ids=np.full(solid.shape,-1,int);self.ids[~solid]=np.arange((~solid).sum());self.n=(~solid).sum()
        self.clear=float(self.wx[inside[0]&~solid.any(axis=0)].sum()/(cells*.3))
        assert abs(self.clear-.75)<1e-9

    def incident(self,f,X=None,Z=None):
        X=self.X if X is None else X;Z=self.Z if Z is None else Z
        k=2*np.pi*f/343;r=np.maximum(np.hypot(X,Z+3),1e-10)
        return hankel1(0,k*r)/abs(hankel1(0,k))

    def solve(self,f,empty=False):
        k=2*np.pi*f/343;solid=np.zeros_like(self.solid) if empty else self.solid
        ids=np.full(solid.shape,-1,int);ids[~solid]=np.arange((~solid).sum());n=int((~solid).sum());wx,wz=self.wx,self.wz
        def stretch(a,lo,hi):
            d=np.maximum(np.maximum(lo-a,a-hi),0)/self.pml
            return 1+1j*46/(k*self.pml)*d**3
        sx=stretch(self.x,-6.75,6.75);sz=stretch(self.z,-3.75,4.75)
        sxf=stretch(self.xe[1:-1],-6.75,6.75);szf=stretch(self.ze[1:-1],-3.75,4.75)
        gx=(wz*sz)[:,None]/(np.diff(self.x)*sxf)[None,:]
        gz=(wx*sx)[None,:]/(np.diff(self.z)*szf)[:,None]
        diagonal=(k*k*(wz*sz)[:,None]*(wx*sx)[None,:])[~solid].copy()
        rows=[];cols=[];vals=[];rhs=np.zeros(n,complex)
        # Same discrete monopole and mesh in the wall and no-wall calculations.
        ix=int(np.searchsorted(self.x,0)-1);iz=int(np.searchsorted(self.z,-3)-1)
        tx=(0-self.x[ix])/(self.x[ix+1]-self.x[ix]);tz=(-3-self.z[iz])/(self.z[iz+1]-self.z[iz])
        for j,wj in [(iz,1-tz),(iz+1,tz)]:
            for i,wi in [(ix,1-tx),(ix+1,tx)]:rhs[ids[j,i]]+=4j*wj*wi/abs(hankel1(0,k))
        def connect(a,b,g,axis):
            ok=(a>=0)&(b>=0);aa=a[ok];bb=b[ok];gg=g[ok]
            rows.extend([aa,bb]);cols.extend([bb,aa]);vals.extend([gg,gg])
            np.add.at(diagonal,aa,-gg);np.add.at(diagonal,bb,-gg)
        connect(ids[:,:-1],ids[:,1:],gx,0);connect(ids[:-1,:],ids[1:,:],gz,1)
        # Homogeneous total-field Dirichlet condition beyond the PML.
        for j in [0,-1]:
            diagonal[ids[:,j]]-=2*wz*sz/(wx[j]*sx[j])
            diagonal[ids[j,:]]-=2*wx*sx/(wz[j]*sz[j])
        ii=np.arange(n);rows.append(ii);cols.append(ii);vals.append(diagonal)
        A=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsc()
        solution=spsolve(A,rhs)
        residual=float(np.linalg.norm(A@solution-rhs)/max(np.linalg.norm(rhs),1e-30))
        field=np.full(ids.shape,np.nan+0j);inc=self.incident(f);field[~solid]=solution
        assert np.isfinite(field[~solid]).all() and residual<1e-7
        return field,inc,residual

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--frequency',type=float);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);mesh=FiniteArray()
    np.savez_compressed(OUT/'mesh.npz',x=mesh.x,z=mesh.z,xe=mesh.xe,ze=mesh.ze,solid=mesh.solid)
    print('Mesh',mesh.solid.shape,'air unknowns',mesh.n,'clear',mesh.clear,flush=True)
    fs=[args.frequency] if args.frequency else FC[FC<=1600]
    for f in fs:
        path=OUT/f'field_{f:g}.npz'
        if path.exists():continue
        start=time.time();p,_,res=mesh.solve(float(f));inc,_,base_res=mesh.solve(float(f),empty=True);np.savez_compressed(path,p=p,incident=inc,residual=res,baseline_residual=base_res)
        print('Solved',f,'Hz',round(time.time()-start,1),'seconds, residual',res,flush=True)
    if args.frequency:return
    weighted=np.zeros(mesh.solid.shape);baseline=np.zeros_like(weighted);tail=np.zeros_like(weighted)
    for f,power,gain in zip(FC,P,G):
        if power==0:continue
        if f<=1600:
            q=np.load(OUT/f'field_{f:g}.npz');weighted+=power*gain*abs(q['p'])**2;baseline+=power*gain*abs(q['incident'])**2
        else:tail+=power*gain*abs(mesh.incident(f))**2
    np.savez_compressed(OUT/'broadband.npz',weighted=weighted,baseline=baseline,tail=tail)
    report=dict(method='Explicit finite-volume 2D total-field Helmholtz, sound-hard cavities, stretched-coordinate PML; identical discrete point source and grid in wall and no-wall solves',
                cells=33,width_m=9.9,depth_m=mesh.depth,intended_height_m=1.8,height_not_modelled=True,clear_fraction=mesh.clear,
                source=dict(x=0,z=-3,normalisation='Unit magnitude at 1 m for each frequency before applying measured band power'),
                mesh=dict(shape=list(mesh.solid.shape),air_unknowns=int(mesh.n),maximum_cell_in_wall_m=mesh.h,maximum_cell_away_m=mesh.far,pml_thickness_m=mesh.pml),
                frequencies_solved=fs.tolist(),weighted_fraction_solved=float((P*G)[FC<=1600].sum()/(P*G).sum()),
                limitations=['2D point source is a 3D line source; no finite-height diffraction or ground reflections',
                             'Coarse mesh, particularly above 1 kHz; not a converged performance estimate',
                             'One frequency at each supplied band centre; no within-band averaging',
                             'Whole-spectrum approximation adds all above-1600-Hz bands as unattenuated incoherent background',
                             'Band powers combine incoherently; pressure snapshots are illustrative single-frequency phases'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(mesh,weighted,baseline,tail)

def plot(mesh,w,b,tail,refined=False):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    norm=float((P*G).sum());eps=1e-15
    fig,axs=plt.subplots(2,2,figsize=(15,11),layout='constrained')
    pressure=np.load(OUT/('fine_1000.npz' if refined else 'field_1000.npz'))
    data=[10*np.log10((b+tail)/norm+eps),10*np.log10((w+tail)/norm+eps),
          10*np.log10((b+tail)/(w+tail)+eps),pressure['p'].real]
    titles=['No wall · weighted level','Finite wall · weighted level (high-frequency tail unchanged)',
            'Approximate insertion loss · positive means quieter','1,000 Hz · instantaneous real pressure']
    for i,(ax,d,title) in enumerate(zip(axs.flat,data,titles)):
        if i and i!=3:d=np.ma.masked_where(mesh.solid,d)
        kw=dict(cmap='magma',vmin=-15,vmax=3) if i<2 else (dict(cmap='RdBu',norm=TwoSlopeNorm(vmin=-8,vcenter=0,vmax=15)) if i==2 else dict(cmap='RdBu_r',vmin=-.8,vmax=.8))
        if i==3 and refined:im=ax.pcolormesh(pressure['x'],pressure['z'],d,shading='nearest',rasterized=True,**kw)
        else:im=ax.pcolormesh(mesh.xe,mesh.ze,d,shading='flat',rasterized=True,**kw)
        if i:ax.contourf(mesh.x,mesh.z,mesh.solid,levels=[.5,1.5],colors=['#30363d'])
        ax.plot(0,-3,'*',color='#ffc400',ms=11,markeredgecolor='black');ax.set(xlim=(-6.5,6.5),ylim=(-3.6,4.5),xlabel='Across wall x (m)',ylabel='Propagation direction z (m)',title=title)
        ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='dB relative to weighted source level at 1 m' if i<2 else ('dB' if i==2 else 'Relative pressure'))
    fig.suptitle('9.9 m wall · 33 explicit cavity cells · 75% clear opening',fontsize=18)
    fig.supxlabel('Horizontal 2D section; source 3 m in front. Intended height 1.8 m is not represented.\n25–1,600 Hz solved (94.9% of weighted input); remaining 5.1% added unchanged. Band-centre sampling.'+('\n630–1,600 Hz use a finer mesh; this remains a screening calculation, not a converged outdoor prediction.' if refined else ' Coarse mesh.'),fontsize=11)
    fig.savefig(OUT/'finite_wall_maps.png',dpi=160);plt.close(fig)
    # Detailed view makes the actual repeated cavity geometry inspectable.
    fig,ax=plt.subplots(figsize=(13,3),layout='constrained');ax.pcolormesh(mesh.xe,mesh.ze,mesh.solid,cmap='Greys',vmin=0,vmax=1,shading='flat');ax.set(xlim=(-.6,.6),ylim=(-.04,mesh.depth+.04),xlabel='Across wall x (m)',ylabel='Depth z (m)',title='Four-cell detail of the explicit computational geometry');ax.set_aspect('equal');fig.savefig(OUT/'geometry_detail.png',dpi=180);plt.close(fig)

if __name__=='__main__':run()
