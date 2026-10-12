"""Initial fixed-geometry smooth-cavity slat simulation; no optimisation."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,csv
from pathlib import Path
import numpy as np
from .benchmark_mesh import segmented_grid,PreparedMesh
OUT=Path(__file__).parent/'results'/'smooth_slat_initial'

def geometry(h,cavities=True):
 x=segmented_grid([0,.08,.083,.098,.102,.117,.120,.2],h)
 z=segmented_grid([-.02,0,.02,.04,.06,.08,.127,.146,.18,.2],h)
 xx=(x[1:]+x[:-1])/2;zz=(z[1:]+z[:-1])/2;X=xx[None,:];Z=zz[:,None]
 s=(X>.08)&(X<.12)&(Z>0)&(Z<.18)
 if cavities:
  s &= ~(((X>.083)&(X<.098)&(Z>.02)&(Z<.127))|((X>.102)&(X<.117)&(Z>.06)&(Z<.146))|((X>=.08)&(X<.083)&(Z>.02)&(Z<.04))|((X>.117)&(X<=.12)&(Z>.06)&(Z<.08)))
 clear=float(np.diff(x)[~s.any(axis=0)].sum()/.2)
 assert clear>=.75-1e-12
 return s,x,z,clear

def main():
 OUT.mkdir(parents=True,exist_ok=True);cache=OUT/'runs.json';rows=json.loads(cache.read_text()) if cache.exists() else []
 def runset(h,fs,cavities):
  s,x,z,clear=geometry(h,cavities);mesh=PreparedMesh(s,x,z,boundary='periodic')
  for f in fs:
   if any(r['h']==h and r['f']==f and r['cavities']==cavities for r in rows):continue
   r=mesh.solve(frequency=f)
   row=dict(h=h,f=float(f),cavities=cavities,T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual,clear=clear,grid=list(s.shape))
   rows.append(row);cache.write_text(json.dumps(rows,indent=2));print(row,flush=True)
   if cavities and h==.001:
    d=np.linspace(0,.25,251)
    np.savez_compressed(OUT/f'field_{f:g}.npz',solid=s,x=x,z=z,p=r.pressure,d=d,up=r.extend(d,upstream=True),down=r.extend(d))
 fs=np.unique(np.r_[np.arange(600,1601,50),np.arange(720,1001,10)])
 runset(.002,fs,True);runset(.002,fs,False)
 coarse=sorted([r for r in rows if r['h']==.002 and r['cavities']],key=lambda r:r['f'])
 best=min(coarse,key=lambda r:r['T'])['f']
 runset(.001,sorted(set([800.,1000.,1250.,best])),True)
 fine=sorted([r for r in rows if r['h']==.001 and r['cavities']],key=lambda r:r['f'])
 # Third mesh at 1 kHz to distinguish a trend from a two-grid coincidence.
 runset(.0005,sorted(set([1000.,best])),True)
 report=dict(status='Initial fixed-geometry screening; not optimised or outdoor validated',geometry=dict(pitch_mm=200,slat_width_mm=40,depth_mm=180,clear_fraction=.8,passage_width_mm=15,lengths_mm=[107,86],mouth_axial_height_mm=20,outer_wall_mm=3,septum_mm=4),coarse_sampled_minimum=best,one_khz=[r for r in rows if r['f']==1000 and r['cavities']],fine_checks=fine,max_energy_error=max(r['energy_error'] for r in rows),assumptions=['Infinite periodic 2D array; normal incidence; c=343 m/s','Rigid lossless walls, open mouths; no insect screen or viscothermal impedance','Periodic transverse boundary, exact discrete modal radiation axial boundaries','Fixed geometry from concept drawing; no phase-gradient or broadband optimisation','80% clear per repeating cell; whole-wall posts/rails excluded'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 with (OUT/'transmission.csv').open('w') as file:
  w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(11,11),layout='constrained',gridspec_kw={'height_ratios':[1.2,.8,1]})
 ax=axs[0]
 for cav,label,color in [(False,'Solid slat, no cavities (2 mm mesh)','#8c9298'),(True,'Two smooth cavities (2 mm mesh)','#128581')]:
  rs=sorted([r for r in rows if r['h']==.002 and r['cavities']==cav],key=lambda r:r['f']);ax.plot([r['f'] for r in rs],[r['T'] for r in rs],'o-',color=color,label=label,markersize=4)
 ax.scatter([r['f'] for r in fine],[r['T'] for r in fine],marker='x',s=55,color='#d57420',label='Cavity design: 1 mm mesh',zorder=5)
 third=[r for r in rows if r['h']==.0005];ax.scatter([r['f'] for r in third],[r['T'] for r in third],marker='+',s=100,color='#733d9a',label='Cavity design: 0.5 mm mesh',zorder=6)
 ax.axvspan(800,1250,color='#f0dba6',alpha=.25);ax.set(xlabel='Frequency (Hz)',ylabel='Transmitted power fraction T',ylim=(-.02,1.04),title='Initial transmission sweep · fixed 80%-clear geometry');ax.legend(fontsize=9);ax.grid(alpha=.2)
 a=np.load(OUT/'field_1000.npz');s=a['solid'];x=a['x'];z=a['z'];p=a['p'];d=a['d'];xc=(x[:-1]+x[1:])/2;zc=(z[:-1]+z[1:])/2
 ax=axs[1];ax.pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#eef5f6','#364450']),shading='flat');ax.set_aspect('equal');ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title='Horizontal section · 40 mm slat / 200 mm pitch · 160 mm clear gap per repeat')
 ax=axs[2];ax.set_facecolor('#364450')
 ax.pcolormesh(z*1000,x*1000,np.ma.masked_where(s,p.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat')
 ax.pcolormesh((zc[0]-d[::-1])*1000,xc*1000,a['up'][::-1].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest')
 im=ax.pcolormesh((zc[-1]+d)*1000,xc*1000,a['down'].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest')
 ax.set_aspect('equal');r1=next(r for r in fine if r['f']==1000)
 ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title=f'Computed pressure at 1,000 Hz · 1 mm mesh · T = {r1["T"]:.4f}, R = {r1["R"]:.4f}')
 fig.colorbar(im,ax=ax,label='Re(p) / incident pressure',shrink=.8)
 fig.suptitle('Smooth-cavity slat: initial acoustic simulation',fontsize=17)
 fig.supxlabel('Rigid, lossless 2D periodic model · normal incidence · screens, posts, finite-height diffraction and weather excluded.\n10 Hz sampling near the main dips; 50 Hz elsewhere. Narrower features can be missed.\nPressure colours saturate outside ±2. This is not a measured wall rating.',fontsize=10)
 fig.savefig(OUT/'initial_simulation.png',dpi=170);plt.close(fig)
 (OUT/'README.md').write_text('''# Smooth-cavity slat initial simulation

Run: `.venv/bin/python -m acoustic_wall.smooth_slat_initial`.

Fixed geometry is the previous drawn concept, translated into a periodic cell: slat x=80–120 mm; depth z=0–180 mm; first cavity x=83–98, z=20–127 mm, mouth through x=80–83, z=20–40 mm; second cavity x=102–117, z=60–146 mm, mouth x=117–120, z=60–80 mm. All rectangle interfaces are exact mesh edges. No dimensions were optimised. Geometric cavity lengths are not end-corrected acoustic lengths.

80% straight-through opening per repeat; final posts/rails must preserve at least 75% overall. Normal plane incidence, rigid lossless 2D Helmholtz, periodic transverse boundaries and discrete outgoing modal axial ports. No insect screens, thermal/viscous losses, structural vibration, finite-height or ground effects. Power transmission includes every propagating outgoing mode. The control is the identical slat with both cavities filled in, on the same mesh.

Coarse sweeps at 50 Hz spacing over 600–1600 Hz, supplemented by 10 Hz samples from 720–1000 Hz, selected 1 mm checks, and a 0.5 mm check at 1000 Hz. Modal continuation visualises exterior pressure without changing the port solution. See runs.json and transmission.csv for all results, report.json for dimensions/assumptions, field_*.npz for complex fields. Cache must be invalidated if geometry/solver changes. These runs are a screening study, not a validated outdoor attenuation prediction.
''')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
