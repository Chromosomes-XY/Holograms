"""Matched-grid comparison: fill the cavities while retaining 75% opening."""
import json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .finite_array_2d import FiniteArray,OUT,FC,P,G

DEST=OUT/'filled_comparison'
REFINED=[630,800,1000,1250,1600]

def main():
    DEST.mkdir(exist_ok=True)
    meshes={False:FiniteArray(filled=True),True:FiniteArray(h=.01,far=.02,filled=True)}
    coarse=meshes[False];weighted=np.zeros(coarse.solid.shape);band_rows=[]
    coordinates=np.stack([coarse.Z.ravel(),coarse.X.ravel()],axis=-1)
    for f,power,gain in zip(FC,P,G):
        if f>1600:continue
        refined=f in REFINED;mesh=meshes[refined];path=DEST/f'field_{f:g}.npz'
        if not path.exists():
            start=time.time();p,_,res=mesh.solve(float(f));np.savez_compressed(path,p=p,residual=res)
            print('Filled',f,'Hz',round(time.time()-start,1),'seconds',flush=True)
        q=np.load(path);a=abs(q['p'])**2
        existing=np.load(OUT/(f'fine_{f:g}.npz' if refined else f'field_{f:g}.npz'))
        area=mesh.wz[:,None]*mesh.wx[None,:];mask=(abs(mesh.X)<4)&(mesh.Z>.7)&(mesh.Z<3.5)
        sums=[float(np.sum(abs(p[mask])**2*area[mask])) for p in [existing['incident'],q['p'],existing['p']]]
        band_rows.append(dict(frequency=float(f),no_wall=sums[0],filled=sums[1],cavities=sums[2],cavity_advantage_db=float(10*np.log10(sums[1]/sums[2]))))
        if refined:a=RegularGridInterpolator((mesh.z,mesh.x),a,method='nearest')(coordinates).reshape(coarse.solid.shape)
        weighted+=power*gain*a
    original=np.load(OUT/'broadband.npz');tail=original['tail'];cavity=original['weighted']+tail;filled=weighted+tail;baseline=original['baseline']+tail
    area=coarse.wz[:,None]*coarse.wx[None,:];region=(abs(coarse.X)<4)&(coarse.Z>.7)&(coarse.Z<3.5)
    sums=[float(np.sum(a[region]*area[region])) for a in [baseline,filled,cavity]]
    report=dict(comparison='All cavity voids filled with rigid material; same external slat envelope, 75% opening, grids, depth, source, boundary treatment, and spectral assumptions',
                clear_fraction=coarse.clear,width_m=9.9,depth_m=coarse.depth,source_x_m=0,source_z_m=-3,
                receiver_region='x=-4..4 m, z=0.7..3.5 m; area-integrated mean-square pressure, not transmitted flux',
                filled_insertion_loss_db=float(10*np.log10(sums[0]/sums[1])),cavity_insertion_loss_db=float(10*np.log10(sums[0]/sums[2])),
                cavity_advantage_db=float(10*np.log10(sums[1]/sums[2])),bands=band_rows,
                limitations=['Horizontal 2D model; no 1.8 m finite-height effects or ground',
                             'Band-centre sampling through 1600 Hz, 94.9% of weighted power; remaining bands added unchanged to both designs',
                             'Five dominant bands use the same finer mesh as the prior cavity simulation; not spatially converged'])
    (DEST/'report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(DEST/'weighted_comparison.npz',filled=filled,cavity=cavity,baseline=baseline)
    plot(coarse,meshes[True],filled,cavity,report)
    print(json.dumps({k:v for k,v in report.items() if k.endswith('_db')},indent=2),flush=True)
    (DEST/'README.md').write_text('''# Filled-rectangle comparison

Run `.venv/bin/python -m acoustic_wall.compare_filled_array` after the finite-wall and dominant-band refinement runs.

The comparison fills every cavity with rigid material while preserving the external slat envelope, 9.9 m array width, 373.4 mm depth, 300 mm pitch and 225 mm clear gap (75%). Interior shared slats are 75 mm wide; the finite array retains the same 37.5 mm half-slats at the two outer ends. This is an open array of solid rectangles, not a completely closed solid wall.

Both geometries use identical grid edges, source position, PML, rigid boundary conditions and frequency sampling. Existing no-wall and cavity fields are reused. Bands through 1600 Hz are explicitly solved; the five dominant bands use the refined mesh. The remaining 5.1% of supplied weighted input is added identically as unattenuated background. Band powers add incoherently. No absolute SPL is inferred.

In the difference maps, positive dB means the cavity array is quieter than the filled rectangles; negative means it is louder locally. Receiver-region comparisons integrate mean-square pressure over x=-4..4 m, z=0.7..3.5 m before taking the dB ratio. They are not transmitted flux or an outdoor wall rating. All prior finite-array limitations apply, including 2D line-source behaviour, no height/ground, band-centre sampling and incomplete mesh convergence.
''')

def plot(coarse,fine,filled,cavity,report):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axs=plt.subplots(2,3,figsize=(19,11),layout='constrained')
    norm=float((P*G).sum());solid_p=np.load(DEST/'field_1000.npz')['p'];cavity_p=np.load(OUT/'fine_1000.npz')['p']
    cavity_coarse=FiniteArray();cavity_fine=FiniteArray(h=.01,far=.02)
    fields=[10*np.log10(filled/norm),10*np.log10(cavity/norm),10*np.log10(filled/cavity),
            solid_p.real,cavity_p.real,10*np.log10(abs(solid_p)**2/abs(cavity_p)**2)]
    titles=['Filled rectangles · weighted sound level','Optimised cavities · weighted sound level','Broadband advantage of cavities',
            'Filled rectangles · 1 kHz pressure','Optimised cavities · 1 kHz pressure','1 kHz advantage of cavities']
    for i,(ax,d,title) in enumerate(zip(axs.flat,fields,titles)):
        mesh=coarse if i<3 else fine;solid=mesh.solid if i%3!=1 else (cavity_coarse.solid if i<3 else cavity_fine.solid)
        kw=dict(cmap='magma',vmin=-15,vmax=3) if i<2 else (dict(cmap='RdBu_r',vmin=-.8,vmax=.8) if i in [3,4] else dict(cmap='RdBu',norm=TwoSlopeNorm(vmin=-12,vcenter=0,vmax=12)))
        im=ax.pcolormesh(mesh.xe,mesh.ze,np.ma.masked_where(solid,d),shading='flat',rasterized=True,**kw)
        ax.contourf(mesh.x,mesh.z,solid,levels=[.5,1.5],colors=['#30363d']);ax.plot(0,-3,'*',color='#ffc400',ms=10,markeredgecolor='black')
        ax.set(xlim=(-6.5,6.5),ylim=(-3.6,4.5),xlabel='Across wall x (m)',ylabel='Propagation z (m)',title=title);ax.set_aspect('equal')
        fig.colorbar(im,ax=ax,label='Relative weighted level (dB)' if i<2 else ('Relative pressure' if i in [3,4] else 'dB · positive = cavities quieter'))
    fig.suptitle('Cavities versus filled rectangular slats · identical 75% opening and 9.9 m width',fontsize=18)
    fig.supxlabel(f'Receiver-region weighted reduction: filled rectangles {report["filled_insertion_loss_db"]:.2f} dB; cavities {report["cavity_insertion_loss_db"]:.2f} dB.\n2D screening only; source 3 m in front. Band-centre sampling through 1,600 Hz; higher-frequency tail added unchanged to both. No over-top diffraction.',fontsize=12)
    fig.savefig(DEST/'comparison_maps.png',dpi=150);plt.close(fig)
    # Common-scale geometry makes the experimental control explicit.
    fig,axs=plt.subplots(1,2,figsize=(12,3),layout='constrained')
    for ax,mesh,title in zip(axs,[coarse,cavity_coarse],['Filled rectangular slats','Optimised cavities']):
        ax.pcolormesh(mesh.xe,mesh.ze,mesh.solid,cmap='Greys',shading='flat');ax.set(xlim=(-.6,.6),ylim=(-.03,.41),xlabel='Across wall x (m)',ylabel='Depth z (m)',title=title);ax.set_aspect('equal')
    fig.savefig(DEST/'geometry_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
