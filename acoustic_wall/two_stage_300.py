"""300 mm cell, two independently dimensioned left/right cavity sets."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from .benchmark_mesh import segmented_grid,PreparedMesh
from .optimise_smooth_pair import F,WEIGHTS
OUT=Path(__file__).parent/'results'/'two_stage_300'
# Four lengths: L1,R1,L2,R2; axial separation after the last first-set cavity.
SEED=[107.,86.,78.,69.,80.]
BOUNDS=[(55,130),(55,130),(45,115),(45,115),(20,160)]

def layout(v):
 l1,r1,l2,r2,gap=v
 second=max(20+l1,35+r1)+gap
 cs=[dict(name='L1',side='left',length=l1,start=20.,mouth=20.,width=22.),dict(name='R1',side='right',length=r1,start=35.,mouth=20.,width=22.),dict(name='L2',side='left',length=l2,start=second,mouth=20.,width=22.),dict(name='R2',side='right',length=r2,start=second+15,mouth=20.,width=22.)]
 return cs,max(c['start']+c['length'] for c in cs)+10

def geometry(cs,depth,h):
 xs=[0,.03,.27,.3];zs=[-.02,0,depth/1000,(depth+20)/1000];rects=[]
 for c in cs:
  a=c['start']/1000;b=a+c['length']/1000;m=c['mouth']/1000;w=c['width']/1000
  assert 0<m<c['length']/1000 and 0<w<=.024 and a>0 and b<depth/1000
  if c['side']=='left':rects.extend([(.027-w,.027,a,b),(.027,.03,a,a+m)])
  else:rects.extend([(.273,.273+w,a,b),(.27,.273,a,a+m)])
  zs.extend([a,a+m,b])
 for a,b,c,d in rects:xs.extend([a,b])
 # Reject intersecting same-side cavities even if generic callers change starts.
 for side in ['left','right']:
  group=sorted([c for c in cs if c['side']==side],key=lambda c:c['start'])
  assert all(b['start']-(a['start']+a['length'])>=3 for a,b in zip(group,group[1:]))
 x=segmented_grid(xs,h);z=segmented_grid(zs,h);X=(x[1:]+x[:-1])[None,:]/2;Z=(z[1:]+z[:-1])[:,None]/2
 solid=((X<.03)|(X>.27))&(Z>0)&(Z<depth/1000)
 for a,b,c,d in rects:solid &= ~((X>a)&(X<b)&(Z>c)&(Z<d))
 clear=float(np.diff(x)[~solid.any(axis=0)].sum()/.3);assert clear>=.8-1e-10
 return solid,x,z,clear

def evaluate(v,fs,h,save=False):
 cs,depth=layout(v);s,x,z,clear=geometry(cs,depth,h);mesh=PreparedMesh(s,x,z,boundary='periodic');rs=[]
 for f in fs:
  r=mesh.solve(frequency=f);rs.append(dict(f=float(f),T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual))
  if save and f==1000:
   d=np.linspace(0,.22,221);np.savez_compressed(OUT/'field_1000.npz',s=s,x=x,z=z,p=r.pressure,d=d,up=r.extend(d,True),down=r.extend(d))
 return rs

def main():
 OUT.mkdir(parents=True,exist_ok=True);hist=[];cache={};log=OUT/'history.jsonl'
 if log.exists():
  hist=[json.loads(s) for s in log.read_text().splitlines()];cache={tuple(r['v']):r for r in hist}
 def objective(v):
  key=tuple(np.round(v,1))
  if key in cache:return cache[key]['score']
  rs=evaluate(key,F,.003);t=np.array([r['T'] for r in rs]);row=dict(v=list(key),score=float(.85*(t@WEIGHTS)+.15*max(t)),weighted_T=float(t@WEIGHTS),responses=rs)
  hist.append(row);cache[key]=row
  with log.open('a') as file:file.write(json.dumps(row)+'\n')
  if row['score']<=min(q['score'] for q in hist):print('Best',len(hist),row['score'],key,flush=True)
  return row['score']
 objective(SEED);rng=np.random.default_rng(30);b=np.array(BOUNDS);pop=rng.uniform(b[:,0],b[:,1],size=(14,5));pop[0]=SEED
 differential_evolution(objective,BOUNDS,init=pop,maxiter=5,seed=30,polish=False,tol=0,atol=0,workers=1)
 finalists=[]
 for q in sorted(hist,key=lambda q:q['score'])[:3]:
  rs=evaluate(q['v'],F,.0015);t=np.array([r['T'] for r in rs]);finalists.append(dict(v=q['v'],score=float(.85*(t@WEIGHTS)+.15*max(t)),weighted_T=float(t@WEIGHTS),responses=rs))
 best=min(finalists,key=lambda q:q['score']);v=best['v'];cs,depth=layout(v);fs=np.arange(650,1401,25)
 sweep=evaluate(v,fs,.0015,True);initial=evaluate(SEED,fs,.0015);checks=evaluate(v,[800,1000,1250],.00075)
 initial_obj=evaluate(SEED,F,.0015)
 report=dict(pitch_mm=300,clear_gap_mm=240,clear_fraction=.8,side_half_body_width_mm=30,depth_mm=depth,cavities=cs,separation_mm=v[4],selected=best,initial_v=SEED,bounds=BOUNDS,evaluations=len(hist),finalists=finalists,sweep=sweep,initial_sweep=initial,checks=checks,initial_objective_responses=initial_obj,objective_frequencies=F.tolist(),objective_weights=WEIGHTS.tolist(),objective='0.85 weighted mean T + 0.15 worst T across nine frequencies; A-weighted measured 800/1000/1250 Hz powers distributed across three sample frequencies each.',varied='Four independent cavity lengths and spacing between sets; depth follows their envelope.',fixed='20 mm mouths; 22 mm passage widths; set-relative mouth offsets 20/35 mm or 0/15 mm. All independently configurable in geometry but fixed for this bounded search.',limitations=['Rigid lossless 2D periodic array at normal incidence','No screens, posts, finite height, thermoviscous effects or outdoor validation','80% cell openness, finished wall must remain >=75%','300 mm period admits additional transmitted diffraction orders above about 1143 Hz; T sums all propagating orders','Initial bounded search, not global optimum or full spectral integration'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report)
 (OUT/'README.md').write_text('''# 300 mm cell with two cavity sets

Run `.venv/bin/python -m acoustic_wall.two_stage_300`.

The central gap is 240 mm (80% true frontal opening); side half-bodies occupy 30 mm each. Periodic repetition joins neighbouring half-bodies into 60 mm slats. The final wall must retain >=75% clear area after frames/posts. Four separate straight cavities form two left/right pairs along the depth. Each cavity has independently configurable length, width, mouth height and start in the geometry builder. This initial bounded optimisation varies the four lengths and the separation between sets. Widths are fixed at 22 mm and mouths at 20 mm. No narrow rib structures are introduced.

The second set starts after the end of both first-set cavities plus the optimised separation. Total depth follows the last cavity end plus a 10 mm back cap; depth is not constrained to the earlier 180 mm. Minimum separation between same-side cavities is explicitly checked. The solver uses exact rectangular interfaces and enforces 80% opening on every evaluated mesh.

Fourteen-member differential evolution, five generations, 3 mm screening mesh; three finalists rechecked at 1.5 mm. Selected design swept at 25 Hz intervals on a 1.5 mm mesh and checked at 0.75 mm for 800, 1000 and 1250 Hz. Coarse search bounds and all dimensions are in report.json. The objective uses nine frequency samples with measured A-weighted band-power weights and a worst-frequency penalty; it is not a full band-integrated outdoor metric. history.jsonl is a task-specific cache; invalidate if model changes.

Normal-incidence rigid lossless 2D periodic acoustics, exact discrete modal ports. Transmission sums all propagating orders, including diffracted orders above the first 300 mm-period cut-on (approximately 1143 Hz). Screens, posts, elastic vibration, finite wall and thermoviscous losses are omitted. This is a preliminary design search, not a validated product prediction.
''')
 print('RESULT',depth,v,checks,flush=True)

def plot(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(12,11),layout='constrained',gridspec_kw={'height_ratios':[1,.95,.9]})
 for key,name,col in [('initial_sweep','Starting four-cavity layout','#959b9f'),('sweep','Selected four-cavity layout','#11857f')]:
  a=r[key];axs[0].plot([q['f'] for q in a],[q['T'] for q in a],label=name,color=col)
 a=r['checks'];axs[0].scatter([q['f'] for q in a],[q['T'] for q in a],marker='x',color='#c57725',label='0.75 mm checks')
 axs[0].axvline(343/.3,color='#808080',ls=':',label='First diffraction-order cut-on ≈1143 Hz');axs[0].set(xlabel='Frequency (Hz)',ylabel='Total power transmission T',ylim=(-.02,1.04));axs[0].grid(alpha=.2);axs[0].legend(fontsize=9)
 a=np.load(OUT/'field_1000.npz');s=a['s'];x=a['x'];z=a['z'];p=a['p'];d=a['d'];xc=(x[1:]+x[:-1])/2;zc=(z[1:]+z[:-1])/2
 ax=axs[1];ax.pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#edf6f7','#35444f']),shading='flat');ax.set_aspect('equal');ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title=f'300 mm pitch · 240 mm clear gap · selected depth {r["depth_mm"]:.1f} mm')
 for c in r['cavities']:ax.text(c['start']+c['length']/2,62 if c['side']=='left' else 230,f'{c["name"]}: {c["length"]:.1f} mm',ha='center',fontsize=9)
 ax=axs[2];ax.set_facecolor('#35444f');ax.pcolormesh(z*1000,x*1000,np.ma.masked_where(s,p.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat');ax.pcolormesh((zc[0]-d[::-1])*1000,xc*1000,a['up'][::-1].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');im=ax.pcolormesh((zc[-1]+d)*1000,xc*1000,a['down'].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');ax.set_aspect('equal');ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title='Computed pressure at 1,000 Hz · 1.5 mm mesh');fig.colorbar(im,ax=ax,label='Re(p) / incident pressure',shrink=.7)
 fig.suptitle('300 mm unit cell · two independently tuned left/right cavity sets',fontsize=16)
 fig.supxlabel('Preliminary lossless 2D periodic model. No screens, posts or finite-wall effects; not a validated noise rating.\n80% opening per repeating cell; ≥75% required across the complete wall. Pressure colour range ±2.',fontsize=10)
 fig.savefig(OUT/'two_stage_result.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
