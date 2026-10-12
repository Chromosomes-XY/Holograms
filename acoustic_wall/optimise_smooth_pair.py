"""Independent left/right smooth resonators around an 80%-clear periodic gap."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from .benchmark_mesh import segmented_grid,PreparedMesh
OUT=Path(__file__).parent/'results'/'smooth_pair_optimised'
F=np.array([750,800,850,940,1000,1060,1175,1250,1325.])
WEIGHTS=np.repeat([.7469*.8328,1,.4803*1.1419],3);WEIGHTS=WEIGHTS/WEIGHTS.sum()
SEED=np.array([86,60,20,15,107,20,20,15.])
BOUNDS=[(50,110),(10,65),(10,30),(8,15)]*2

def geometry(v,h):
 ll,zl,ml,wl,lr,zr,mr,wr=np.array(v)/1000
 assert min(v)>0 and ml<ll and mr<lr and max(zl+ll,zr+lr)<.1771
 # Each 20 mm side occupies half a shared 40 mm slat. Exterior wall is 2 mm;
 # mouths cross a 3 mm inner wall; widths <=15 mm retain >=2 mm seam wall.
 x=segmented_grid([0,.017-wl,.017,.020,.180,.183,.183+wr,.2],h)
 z=segmented_grid([-.02,0,zl,zl+ml,zl+ll,zr,zr+mr,zr+lr,.180,.2],h)
 X=(x[1:]+x[:-1])[None,:]/2;Z=(z[1:]+z[:-1])[:,None]/2
 solid=((X<.020)|(X>.180))&(Z>0)&(Z<.180)
 airL=((X>.017-wl)&(X<.017)&(Z>zl)&(Z<zl+ll))|((X>=.017)&(X<.020)&(Z>zl)&(Z<zl+ml))
 airR=((X>.183)&(X<.183+wr)&(Z>zr)&(Z<zr+lr))|((X>.180)&(X<=.183)&(Z>zr)&(Z<zr+mr))
 solid &= ~(airL|airR)
 clear=np.diff(x)[~solid.any(axis=0)].sum()/.2
 assert clear>=.8-1e-10
 return solid,x,z

def evaluate(v,fs,h,fields=False):
 s,x,z=geometry(v,h);mesh=PreparedMesh(s,x,z,boundary='periodic');rs=[]
 for f in fs:
  r=mesh.solve(frequency=f);rs.append(dict(f=float(f),T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual))
  if fields and f==1000:
   d=np.linspace(0,.22,221);np.savez_compressed(OUT/'field_1000.npz',solid=s,x=x,z=z,p=r.pressure,d=d,up=r.extend(d,True),down=r.extend(d))
 return rs

def main():
 OUT.mkdir(parents=True,exist_ok=True);history=[];cache={}
 log=OUT/'history.jsonl'
 if log.exists():
  history=[json.loads(line) for line in log.read_text().splitlines()]
  cache={tuple(r['dimensions_mm']):r for r in history}
 def objective(v):
  key=tuple(np.round(v,1))
  if key in cache:return cache[key]['score']
  rs=evaluate(key,F,.0025);t=np.array([r['T'] for r in rs]);score=float(.85*(t@WEIGHTS)+.15*max(t))
  r=dict(dimensions_mm=list(key),score=score,weighted_T=float(t@WEIGHTS),responses=rs);cache[key]=r;history.append(r)
  with log.open('a') as file:file.write(json.dumps(r)+'\n')
  if score<=min(a['score'] for a in history):print('Best',len(history),round(score,4),key,flush=True)
  return score
 objective(SEED)
 rng=np.random.default_rng(29);bounds=np.array(BOUNDS);pop=rng.uniform(bounds[:,0],bounds[:,1],size=(16,8));pop[0]=SEED
 differential_evolution(objective,BOUNDS,init=pop,maxiter=6,polish=False,seed=29,workers=1,tol=0,atol=0)
 # Select among three coarse finalists by the same objective on a finer mesh.
 shortlist=sorted(history,key=lambda r:r['score'])[:3];fine=[]
 for candidate in shortlist:
  v=candidate['dimensions_mm'];rs=evaluate(v,F,.001);t=np.array([r['T'] for r in rs]);fine.append(dict(dimensions_mm=v,score=float(.85*(t@WEIGHTS)+.15*max(t)),weighted_T=float(t@WEIGHTS),responses=rs))
 best=min(fine,key=lambda r:r['score']);v=best['dimensions_mm'];print('Selected',best,flush=True)
 fs=np.arange(650,1401,25)
 initial=evaluate(SEED,fs,.001);sweep=evaluate(v,fs,.001,True)
 checks=evaluate(v,[800,1000,1250],.0005)
 baseline=evaluate(SEED,F,.001)
 report=dict(dimensions_order=['left_length','left_mouth_start','left_mouth_height','left_cavity_width','right_length','right_mouth_start','right_mouth_height','right_cavity_width'],units='mm',initial_dimensions_mm=SEED.tolist(),selected=best,finalists=fine,baseline_objective_responses=baseline,coarse_evaluations=len(history),bounds_mm=BOUNDS,objective_frequencies_hz=F.tolist(),objective_weights=WEIGHTS.tolist(),objective='0.85 weighted mean T + 0.15 worst T across nine frequencies. Weights use measured A-weighted 800/1000/1250 Hz band powers, spread over three frequency samples each; not exact band integration.',initial_sweep=initial,optimised_sweep=sweep,half_mm_checks=checks,clear_fraction=.8,pitch_mm=200,depth_mm=180,status='Bounded initial optimisation; not global optimum, full mesh convergence or outdoor validation',limitations=['Rigid lossless 2D periodic at normal incidence','Screens, supports, finite wall and losses excluded','Final support layout must preserve >=75% whole-wall opening','No asserted spoof-surface-wave or paper phase-gradient replication'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 plot(report)
 (OUT/'README.md').write_text('''# Independent left/right smooth resonator optimisation

Run `.venv/bin/python -m acoustic_wall.optimise_smooth_pair`.

One 200 mm periodic cell has a left side body x=0–20 mm, a right side body x=180–200 mm, and a 160 mm unobstructed central gap. Each side has one smooth rectangular cavity opening sideways into this same gap. Periodic repetition joins half-bodies into 40 mm slats. The two cavities are independently dimensioned and coupled through the full Helmholtz solution, not solved as isolated resonators. Translating the previous centre-slat geometry into this convention preserves the baseline (15 mm passages and a combined 4 mm separator across the periodic seam).

Each side independently varies cavity length (50–110 mm), mouth start (10–65 mm from front), mouth height (10–30 mm), and cavity width (8–15 mm). Wall depth remains 180 mm, inner mouth wall 3 mm. All rectangle edges align with the numerical grid. At least 2 mm outer wall remains on each periodic half; together these make a >=4 mm separator. No internal ribs. 80% clear opening enforced per cell, leaving an allowance for whole-wall supports; final whole-wall opening must remain >=75%.

Fixed seed, 16-member differential evolution and six generations on a 2.5 mm maximum mesh. Three finalists checked at 1 mm; best fine objective selected. Subsequent 25 Hz sweep at 1 mm and selected 0.5 mm checks. Objective weights come from the supplied A-weighted measured band powers, distributed over three frequencies around each of 800/1000/1250 Hz, with a worst-frequency penalty. This is a finite sampled screening objective, not an exact broadband integral. See report.json for every parameter and result. history.jsonl is the evaluation cache; invalidate it if the model changes.

Rigid lossless normal-incidence 2D periodic pressure acoustics; outgoing modal axial ports. Screens, losses, posts and finite-height effects absent. No claims of global optimality or validated yard noise reduction.
''')

def plot(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(11,11),layout='constrained',gridspec_kw={'height_ratios':[1.1,1,1]})
 for key,label,c in [('initial_sweep','Initial dimensions (1 mm mesh)','#91979b'),('optimised_sweep','Optimised left/right pair (1 mm mesh)','#13857f')]:
  a=r[key];axs[0].plot([q['f'] for q in a],[q['T'] for q in a],color=c,label=label)
 a=r['half_mm_checks'];axs[0].scatter([q['f'] for q in a],[q['T'] for q in a],marker='x',color='#cf7422',s=60,label='0.5 mm checks')
 axs[0].set(xlabel='Frequency (Hz)',ylabel='Power transmission T',ylim=(-.02,1.04),title='Independent left/right dimensions · 80% central opening');axs[0].legend();axs[0].grid(alpha=.2)
 a=np.load(OUT/'field_1000.npz');x=a['x'];z=a['z'];s=a['solid'];p=a['p'];d=a['d'];xc=(x[1:]+x[:-1])/2;zc=(z[1:]+z[:-1])/2
 axs[1].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#eef6f7','#374451']),shading='flat');axs[1].set_aspect('equal');axs[1].set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title='Selected shape: left below, right above; clear gap between them')
 v=r['selected']['dimensions_mm'];axs[1].text(210,35,f'LEFT\nLength {v[0]:.1f} mm\nMouth start {v[1]:.1f} mm\nMouth height {v[2]:.1f} mm\nCavity width {v[3]:.1f} mm',fontsize=9);axs[1].text(-120,115,f'RIGHT\nLength {v[4]:.1f} mm\nMouth start {v[5]:.1f} mm\nMouth height {v[6]:.1f} mm\nCavity width {v[7]:.1f} mm',fontsize=9);axs[1].set_xlim(-125,320)
 ax=axs[2];ax.set_facecolor('#374451');ax.pcolormesh(z*1000,x*1000,np.ma.masked_where(s,p.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat')
 ax.pcolormesh((zc[0]-d[::-1])*1000,xc*1000,a['up'][::-1].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');im=ax.pcolormesh((zc[-1]+d)*1000,xc*1000,a['down'].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');ax.set_aspect('equal');ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title='Computed pressure at 1,000 Hz · 1 mm mesh');fig.colorbar(im,ax=ax,label='Re(p) / incident pressure',shrink=.7)
 fig.suptitle('Smooth resonators: independent left/right optimisation',fontsize=17)
 fig.supxlabel('Initial bounded search, not a validated wall design. Rigid lossless 2D periodic model, normal incidence.\nNo insect screens, posts or finite-height effects. 80% cell opening; ≥75% required across the finished wall.',fontsize=10)
 fig.savefig(OUT/'optimised_pair.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
