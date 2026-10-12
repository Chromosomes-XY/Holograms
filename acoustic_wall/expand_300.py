"""Expanded 75%-open, four-cavity optimisation with reproducible cached solves."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from .benchmark_mesh import segmented_grid,PreparedMesh
OUT=Path(__file__).parent/'results'/'expanded_300_75pct'
# L1,R1,L2,R2 each length,width,mouth; then front-left,front-right,gap-left,gap-right.
BOUNDS=[(50,140),(14,31.5),(10,40)]*4+[(10,70),(10,70),(10,150),(10,150)]
F=np.array([f*a for f in [800,1000,1250] for a in [.91,.955,1,1.045,1.09]])
W=np.repeat([.7469*.8328,1,.4803*1.1419],5);W/=W.sum()

def layout(v):
 cs=[];starts=[v[12],v[13],v[12]+v[0]+v[14],v[13]+v[3]+v[15]]
 for i in range(4):
  L,w,m=v[i*3:i*3+3];cs.append(dict(name=['L1','R1','L2','R2'][i],side='left' if i%2==0 else 'right',length=L,width=w,mouth=m,start=starts[i]))
 return cs,max(c['start']+c['length'] for c in cs)+10

def geometry(v,h):
 cs,depth=layout(v);xs=[0,.0375,.2625,.3];zs=[-.02,0,depth/1000,(depth+20)/1000];rects=[]
 for c in cs:
  a=c['start']/1000;b=a+c['length']/1000;m=c['mouth']/1000;w=c['width']/1000
  assert 0<m<c['length']/1000 and .014<=w<=.0315+1e-12
  if c['side']=='left':rects.extend([(.0345-w,.0345,a,b),(.0345,.0375,a,a+m)])
  else:rects.extend([(.2655,.2655+w,a,b),(.2625,.2655,a,a+m)])
  zs.extend([a,a+m,b])
 for a,b,c,d in rects:xs.extend([a,b])
 for side in ['left','right']:
  a,b=[c for c in cs if c['side']==side];assert b['start']-a['start']-a['length']>=9.99
 x=segmented_grid(xs,h);z=segmented_grid(zs,h);X=(x[1:]+x[:-1])[None,:]/2;Z=(z[1:]+z[:-1])[:,None]/2
 s=((X<.0375)|(X>.2625))&(Z>0)&(Z<depth/1000)
 for a,b,c,d in rects:s &= ~((X>a)&(X<b)&(Z>c)&(Z<d))
 clear=np.diff(x)[~s.any(axis=0)].sum()/.3;assert clear>=.75-1e-10
 return s,x,z

def evaluate(v,fs,h,field=False):
 key=hashlib.sha256(json.dumps([list(v),list(map(float,fs)),h]).encode()).hexdigest()[:20]
 path=OUT/'cache'/f'{key}.json'
 if path.exists() and not field:return json.loads(path.read_text())
 s,x,z=geometry(v,h);mesh=PreparedMesh(s,x,z,boundary='periodic');rs=[]
 for f in fs:
  r=mesh.solve(frequency=f);rs.append(dict(f=float(f),T=r.transmission,R=r.reflection,error=r.balance_error,residual=r.residual))
  if field and f==1000:
   d=np.linspace(0,.2,201);np.savez_compressed(OUT/'field_1000.npz',s=s,x=x,z=z,p=r.pressure,d=d,up=r.extend(d,True),down=r.extend(d))
 path.write_text(json.dumps(rs));return rs

def score(rs):
 t=np.array([q['T'] for q in rs]);return float(.85*(t@W)+.15*max(t))

def main():
 (OUT/'cache').mkdir(parents=True,exist_ok=True);hist=[];cache={};path=OUT/'history.jsonl'
 if path.exists():
  hist=[json.loads(s) for s in path.read_text().splitlines()];cache={tuple(q['v']):q for q in hist}
 old=json.loads((Path(__file__).parent/'results/two_stage_300/report.json').read_text());cs=old['cavities']
 seed=[]
 for c in cs:seed.extend([c['length'],c['width'],c['mouth']])
 seed.extend([cs[0]['start'],cs[1]['start'],cs[2]['start']-cs[0]['start']-cs[0]['length'],cs[3]['start']-cs[1]['start']-cs[1]['length']])
 def objective(v):
  key=tuple(np.round(v,1))
  if key in cache:return cache[key]['score']
  rs=evaluate(key,F,.003);row=dict(v=list(key),score=score(rs),weighted_T=float(np.array([q['T'] for q in rs])@W),responses=rs);cache[key]=row;hist.append(row)
  with path.open('a') as file:file.write(json.dumps(row)+'\n')
  if row['score']<=min(q['score'] for q in hist):print('Best',len(hist),round(row['score'],5),flush=True)
  return row['score']
 rng=np.random.default_rng(301);b=np.array(BOUNDS);pop=rng.uniform(b[:,0],b[:,1],size=(24,16));pop[0]=seed
 for i in range(1,12):pop[i]=np.clip(np.array(seed)+rng.normal(0,1,16)*(b[:,1]-b[:,0])*.12,b[:,0],b[:,1])
 differential_evolution(objective,BOUNDS,init=pop,maxiter=9,seed=301,polish=False,tol=0,atol=0,workers=1,mutation=(.4,1),recombination=.8)
 # Re-rank coarse finalists and the seed using shifted and original frequency nodes.
 finalists=sorted(hist,key=lambda q:q['score'])[:4];fine=[]
 dense=np.unique(np.r_[np.arange(700,1401,10),F]);weights=np.interp(dense,[700,800,1000,1250,1400],[.5*.622,.622,1,.5485,.25*.5485]);weights/=weights.sum()
 for q in finalists+[dict(v=seed)]:
  rs=evaluate(q['v'],dense,.0015);t=np.array([a['T'] for a in rs]);fine.append(dict(v=q['v'],dense_score=float(.85*t@weights+.15*max(t)),dense_weighted_T=float(t@weights),responses=rs))
 best=min(fine,key=lambda q:q['dense_score']);v=best['v'];print('Selected fine score',best['dense_score'],flush=True)
 checks=evaluate(v,np.unique(np.r_[F,[800,1000,1250]]),.00075);evaluate(v,[1000],.0015,True)
 cavities,depth=layout(v)
 # Previous 80%-open model on the identical dense frequency grid, for a fair comparison.
 from .two_stage_300 import evaluate as old_evaluate
 comparison_file=OUT/'previous_dense.json'
 if comparison_file.exists():previous=json.loads(comparison_file.read_text())
 else:
  previous=old_evaluate(old['selected']['v'],dense,.0015);comparison_file.write_text(json.dumps(previous))
 report=dict(status='Further bounded screening optimisation, not validated outdoor performance',pitch_mm=300,blocked_width_mm=75,clear_gap_mm=225,clear_fraction=.75,depth_mm=depth,cavities=cavities,selected=best,finalists=fine,half_resolution_checks=checks,previous_80pct_sweep=previous,previous_dense_weighted_T=float(np.array([q['T'] for q in previous])@weights),dense_frequencies=dense.tolist(),dense_weights=weights.tolist(),seed_75pct=seed,coarse_candidates=len(hist),bounds=BOUNDS,variable_order='L1,R1,L2,R2: length,width,mouth; then front L/R and inter-cavity gap L/R',search_frequencies=F.tolist(),search_weights=W.tolist(),selection='Finalists and unchanged seed re-ranked on 10 Hz grid plus search nodes using interpolated weighted spectrum and worst-frequency penalty. This is a screening weighting, not exact measured band-power integration.',limitations=['Normal-incidence rigid lossless 2D periodic model','No screens, absorption, elasticity or finite wall','No remaining frontal blockage budget for supports outside the modelled opaque area','All outgoing propagating orders included','Limited stochastic search, not global optimum'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report)
 print('RESULT',depth,cavities,[(q['f'],q['T']) for q in checks if q['f'] in [800,1000,1250]],flush=True)
 (OUT/'README.md').write_text('''# Expanded 300 mm, 75%-open optimisation

Run `.venv/bin/python -m acoustic_wall.expand_300`.

300 mm periodic pitch; opaque half-bodies 37.5 mm each; 225 mm central gap (75%). Four smooth cavities independently vary length, width and mouth height, with independent first-mouth positions and same-side inter-cavity gaps: 16 variables. Minimum cavity width 14 mm, mouth height 10 mm, inner wall 3 mm, outer half-wall at least 3 mm, same-side partition at least 10 mm. Total depth follows the longest side plus a 10 mm cap. Supports must lie within the opaque budget: no further blockage allowance remains.

24-member differential evolution, nine generations, deterministic seed and 3 mm mesh, with cached evaluations. Objective: 85% measured-spectrum-weighted mean transmission plus 15% worst transmission, at 15 samples around the 800, 1000 and 1250 Hz bands. Four finalists and the unchanged 75% seed are re-ranked on a 10 Hz grid plus training nodes at 1.5 mm. This dense selection uses an explicitly interpolated weighting envelope, not an exact band integration. Checks at 0.75 mm use the original 15 nodes. A fair previous-80%-open comparison uses the same dense grid and 1.5 mm mesh. Cache is model-specific; invalidate on model changes.

Rigid lossless 2D periodic pressure acoustics with exact discrete modal radiation ports. All propagating diffraction orders included. No screens, losses, structural vibration, finite height or ground. Numerical mesh checks do not establish outdoor validity. See report.json for all geometry and comparison data; history.jsonl for the search and cache/ for individual solves.
''')

def plot(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(12,11),layout='constrained')
 for data,label,col in [(r['previous_80pct_sweep'],'Previous 80%-open design','#93999e'),(r['selected']['responses'],'Expanded 75%-open design','#15857e')]:axs[0].plot([q['f'] for q in data],[q['T'] for q in data],label=label,color=col)
 a=r['half_resolution_checks'];axs[0].scatter([q['f'] for q in a],[q['T'] for q in a],marker='x',s=25,color='#cc7827',label='0.75 mm checks');axs[0].set(xlabel='Frequency (Hz)',ylabel='Total power transmission T',ylim=(-.02,1.04));axs[0].legend();axs[0].grid(alpha=.2)
 a=np.load(OUT/'field_1000.npz');s=a['s'];x=a['x'];z=a['z'];p=a['p'];d=a['d'];xc=(x[1:]+x[:-1])/2;zc=(z[1:]+z[:-1])/2
 axs[1].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#eff6f7','#354450']),shading='flat');axs[1].set_aspect('equal');axs[1].set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title=f'225 mm clear gap / 300 mm pitch · depth {r["depth_mm"]:.1f} mm')
 for c in r['cavities']:axs[1].text(c['start']+c['length']/2,75 if c['side']=='left' else 220,c['name'],ha='center')
 ax=axs[2];ax.set_facecolor('#354450');ax.pcolormesh(z*1000,x*1000,np.ma.masked_where(s,p.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat');ax.pcolormesh((zc[0]-d[::-1])*1000,xc*1000,a['up'][::-1].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');im=ax.pcolormesh((zc[-1]+d)*1000,xc*1000,a['down'].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest');ax.set_aspect('equal');ax.set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title='Pressure at 1,000 Hz · 1.5 mm mesh');fig.colorbar(im,ax=ax,label='Re(p) / incident pressure',shrink=.7)
 fig.suptitle('Further optimisation · four independent smooth cavities · 75% clear',fontsize=16)
 fig.supxlabel('Lossless normal-incidence 2D periodic screening; not an outdoor noise rating.\nNo screens or finite-wall effects. Supports must fit inside the 25% opaque area. Pressure colours clipped at ±2.',fontsize=10)
 fig.savefig(OUT/'expanded_result.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
