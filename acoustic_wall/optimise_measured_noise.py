"""Optimise remaining A-weighted measured band power, not tone transmission."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,hashlib,csv
from pathlib import Path
import numpy as np
from .expand_300 import geometry,layout,BOUNDS
from .benchmark_mesh import PreparedMesh
from .yard_spectrum import DATA
OUT=Path(__file__).parent/'results'/'measured_noise_optimised'
INPUT=np.array([[float(v) for v in line.split()] for line in DATA.splitlines()]);FC,P,G=INPUT.T;POWER=P*G;WEIGHT=POWER/POWER.sum()
# Contiguous log-midpoint edges accommodate the rounded nominal band centres.
EDGES=np.r_[FC[0]/2**(1/6),np.sqrt(FC[:-1]*FC[1:]),FC[-1]*2**(1/6)]

def integrate(v,n=5,level='search',cutoff=None,angle=0):
 key=hashlib.sha256(json.dumps([list(v),n,level,angle, 'fine_uniform_1p5_v2' if level!='search' else 'search_v1']).encode()).hexdigest()[:20];path=OUT/'cache'/f'{key}.json'
 if path.exists():return json.loads(path.read_text())
 nodes,weights=np.polynomial.legendre.leggauss(n);weights=weights/2
 meshes={};rows=[];partial=0.;used=0.
 for i in np.argsort(-WEIGHT):
  if POWER[i]==0:
   rows.append(dict(index=int(i),f=float(FC[i]),T=None,after=0.,frequencies=[],transmission=[],errors=[],residuals=[]));continue
  lo,hi=EDGES[i:i+2];fs=(lo+hi)/2+(hi-lo)/2*nodes
  h=(.003 if hi<=9000 else .0015) if level=='search' else .0015
  if h not in meshes:
   s,x,z=geometry(v,h);meshes[h]=(PreparedMesh(s,x,z,boundary='periodic'),s,x,z)
  mesh,s,x,z=meshes[h];ts=[];errors=[];residuals=[]
  for f in fs:
   if angle:
    from .bloch_check import solve_bloch
    q=solve_bloch(s,x,z,float(f),angle);t=q['T'];err=q['energy_error'];res=q['residual']
   else:
    q=mesh.solve(frequency=f);t=q.transmission;err=q.balance_error;res=q.residual
   assert np.isfinite(t) and -1e-6<=t<=1+1e-6 and err<1e-6
   ts.append(t);errors.append(err);residuals.append(res)
  average=float(np.array(ts)@weights);partial+=WEIGHT[i]*average;used+=WEIGHT[i]
  rows.append(dict(index=int(i),f=float(FC[i]),T=average,after=float(POWER[i]*average),frequencies=fs.tolist(),transmission=ts,errors=errors,residuals=residuals))
  # Safe rejection: all uncomputed band contributions are nonnegative.
  if cutoff is not None and partial>=cutoff:
   return dict(v=list(v),pruned=True,lower_bound=float(partial),evaluated_weight=float(used),bands=rows)
 rows.sort(key=lambda r:r['index']);r=dict(v=list(v),pruned=False,J=float(partial),reduction_db=float(-10*np.log10(partial)),n=n,level=level,angle=angle,bands=rows)
 path.write_text(json.dumps(r));return r

def main():
 (OUT/'cache').mkdir(parents=True,exist_ok=True)
 prior=json.loads((Path(__file__).parent/'results/expanded_300_75pct/report.json').read_text());seed=np.array(prior['selected']['v']);b=np.array(BOUNDS)
 # Reproducible local stochastic refinement around the best previous design.
 saved_history=OUT/'search_history.json'
 if saved_history.exists() and len(json.loads(saved_history.read_text()))>=49:
  hist=json.loads(saved_history.read_text());best=min([q for q in hist if not q['pruned']],key=lambda q:q['J']);print('Resuming validation of completed 49-candidate search',flush=True)
 else:
  rng=np.random.default_rng(731);hist=[];best=integrate(seed);hist.append(best);print('Initial all-band J',best['J'],flush=True)
  for step in range(48):
   centre=np.array(best['v']);scale=.14*(1-step/60)
   if step%4==0:
    v=centre.copy();j=(step//4)%16;v[j]+=rng.choice([-1,1])*scale*(b[j,1]-b[j,0])
   else:v=centre+rng.normal(0,scale,16)*(b[:,1]-b[:,0])
   v=np.round(np.clip(v,b[:,0],b[:,1]),1);result=integrate(v,cutoff=best['J']);hist.append(result)
   if not result['pruned'] and result['J']<best['J']:best=result;print('Improved',step+1,best['J'],flush=True)
   (OUT/'search_history.json').write_text(json.dumps(hist));(OUT/'best_search.json').write_text(json.dumps(best,indent=2))
   if (step+1)%8==0:print('Candidates',step+1,'best',best['J'],flush=True)
 # Frequency refinement first, then geometry refinement. Keep the starting design eligible.
 completed=sorted([q for q in hist if not q['pruned']],key=lambda q:q['J']);vs=[]
 for q in completed[:3]+[hist[0]]:
  if q['v'] not in vs:vs.append(q['v'])
 finalists=[integrate(v,9,'fine') for v in vs];selected=min(finalists,key=lambda q:q['J']);before=next(q for q in finalists if q['v']==seed.tolist())
 # Independent higher-order quadrature check of the selected geometry and baseline.
 selected17=integrate(selected['v'],17,'fine');before17=integrate(seed,17,'fine')
 if before17['J']<selected17['J']:selected=before;selected17=before17
 cavities,depth=layout(selected['v'])
 report=dict(objective='sum measured_band_power * supplied_ear_gain * band_average_T / sum measured_band_power * supplied_ear_gain',input_table=INPUT.tolist(),band_edges_hz=EDGES.tolist(),within_band_assumption='Uniform power per Hz; contiguous edges are geometric midpoints of adjacent rounded nominal centres, endpoint ratios 2^(1/6). Ear gain held at supplied band value.',clear_fraction=.75,pitch_mm=300,clear_gap_mm=225,depth_mm=depth,cavities=cavities,search_candidates=len(hist),search_integration_order=5,finalists=finalists,baseline=before17,selected=selected17,quadrature_check=dict(selected_9=selected['J'],selected_17=selected17['J'],baseline_9=before['J'],baseline_17=before17['J']),high_frequency_unresolved_mesh_J_bound=float(WEIGHT[EDGES[1:]>9000].sum()),angles='Optimised for normal incidence only. No assumed directional energy weights.',limitations=['Rigid lossless 2D periodic model; no screens, finite-wall effects, supports or thermoviscous loss','All nonzero supplied bands included; 20 kHz band is zero input power and contributes zero without solving','Limited local optimisation, not global optimum','Band quadrature convergence and spatial resolution are finite; narrow resonances may remain unresolved','75% opening leaves no extra blockage allowance for supports'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(report)
 print('RESULT baseline',before17['J'],'selected',selected17['J'],'dB',selected17['reduction_db'],flush=True)
 (OUT/'README.md').write_text('''# Measured-noise optimisation

Run `.venv/bin/python -m acoustic_wall.optimise_measured_noise`.

Objective is the total remaining supplied A-weighted band power, divided by its unattenuated sum. All 30 supplied rows enter the accounting. The 20 kHz band has zero measured power, so its contribution is exactly zero and no transmission is invented. Gain is a power gain; transmission is a power ratio and is not squared. Band powers are not multiplied by bandwidth again.

Within each nominal one-third-octave band, assume uniform power per Hz. To accommodate rounded nominal centre labels without overlapping or missing intervals, interior edges are geometric means of adjacent centres; outer edges use centre × 2^(±1/6). Supplied ear gain is held constant within each band. Gauss–Legendre quadrature calculates a band average, then multiplies by the measured band power. Five points per band in the search; nine-point fine-mesh finalist selection; seventeen-point independent quadrature check for the winner and the starting geometry. Mesh maxima: search 3 mm below 9 kHz upper-band edge, otherwise 1.5 mm; verification 1.5 mm throughout. The above-9-kHz bands together carry only 0.0504% of weighted input power: even completely unresolved transmission there changes J by at most 0.000504, so they remain at 1.5 mm rather than receiving an extra spatial refinement. No frequency violates the approximately 11-points-per-wavelength criterion. All outgoing propagating modes are included.

48 deterministic local proposals vary the existing 16 parameters, with the previous design retained as baseline. A candidate may be rejected early only when its accumulated nonnegative band contributions already exceed the incumbent full objective; rejected candidates cannot replace the incumbent. This does not remove weak bands from any accepted full score. Reproducible cached evaluations and complete/pruned search history are saved. Invalidate caches if model changes.

Geometry keeps 300 mm pitch, 225 mm clear opening and four independent smooth cavities. No remaining support blockage allowance. Normal-incidence optimisation only; angles must be reported separately until an incident directional power distribution is specified. This lossless periodic 2D study is not a measured yard attenuation rating.
''')

def plot(r):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(12,11),layout='constrained',gridspec_kw={'height_ratios':[1,1.2,.9]})
 for q,label,col in [(r['baseline'],'Previous geometry','#8d969e'),(r['selected'],'Noise-profile optimised','#13867e')]:
  ts=np.array([b['T'] if b['T'] is not None else np.nan for b in q['bands']]);axs[0].semilogx(FC,ts,'o-',label=label,color=col,markersize=3);axs[1].semilogx(FC,[b['after'] for b in q['bands']],'o-',label='After '+label.lower(),color=col,markersize=3)
 axs[1].semilogx(FC,POWER,'o-',color='#ce7924',label='Before wall: measured × ear gain',markersize=3)
 for ax in axs[:2]:ax.set_xlim(25,20000);ax.grid(alpha=.2);ax.legend(fontsize=9);ax.set_xlabel('Nominal band centre (Hz)')
 axs[0].set(ylabel='Band-averaged power transmission',ylim=(-.02,1.04),title=r.get('integration_label','Full supplied spectrum · 17-point integration within each nonzero band'))
 axs[1].set(ylabel='Weighted band power, common normalisation',title=f'Estimated remaining weighted power: {r["baseline"]["J"]:.1%} → {r["selected"]["J"]:.1%}')
 s,x,z=geometry(r['selected']['v'],.0015);axs[2].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#eff6f7','#354450']),shading='flat');axs[2].set_aspect('equal');axs[2].set(xlabel='Depth z (mm) →',ylabel='Transverse x (mm)',title=f'75% clear opening · 300 mm pitch · depth {r["depth_mm"]:.1f} mm')
 fig.suptitle('Optimisation against the measured noise spectrum',fontsize=17)
 fig.supxlabel('Normal-incidence lossless 2D periodic screening, not an outdoor noise rating.\nUniform power/Hz assumed within each band. No extra normalisation after attenuation; zero-power 20 kHz band is not simulated.',fontsize=10)
 fig.savefig(OUT/'noise_profile_result.png',dpi=170);plt.close(fig)
 with (OUT/'weighted_spectrum.csv').open('w') as file:
  w=csv.writer(file);w.writerow(['frequency_hz','band_low','band_high','measured_power','ear_gain','weighted_before','T_previous','T_selected','weighted_after_previous','weighted_after_selected'])
  for i,(a,b) in enumerate(zip(r['baseline']['bands'],r['selected']['bands'])):w.writerow([FC[i],EDGES[i],EDGES[i+1],P[i],G[i],POWER[i],a['T'],b['T'],a['after'],b['after']])
if __name__=='__main__':main()
