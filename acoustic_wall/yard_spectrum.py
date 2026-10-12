"""Band-centre screening of the saved 75%-clear candidate against user data."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,csv
from pathlib import Path
import numpy as np
from .geometry import Geometry,rasterize
from .benchmark_mesh import PreparedMesh
OUT=Path(__file__).parent/'results'/'yard_spectrum'
DATA='''25 .0008 .00003
31.5 .0033 .00011
40 .0423 .00035
50 .1437 .00093
63 .9191 .0024
80 .4883 .0058
100 .3941 .0122
125 .2758 .024
160 .264 .0474
200 .5177 .0823
250 .4341 .1357
315 .2418 .2166
400 .2167 .3331
500 .2268 .4734
630 .2935 .6444
800 .7469 .8328
1000 1 1
1250 .4803 1.1419
1600 .1268 1.2569
2000 .0546 1.3188
2500 .0287 1.34
3150 .0176 1.3187
4000 .0084 1.2484
5000 .006 1.1362
6300 .0031 .9737
8000 .0012 .7679
10000 .0016 .5634
12500 .0015 .3755
16000 .0003 .2135
20000 0 .1162'''

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).parent/'results'/'phase_guided_final'/'report.json'
 previous=json.loads(source.read_text());g=Geometry(**previous['geometry'])
 inputs=np.array([[float(v) for v in line.split()] for line in DATA.splitlines()]);freq,power,gain=inputs.T
 cache=OUT/'solves.json';records=json.loads(cache.read_text()) if cache.exists() else []
 for h,fs in [(.001,freq),(.0005,[800,1000,1250])]:
  s,dx,dz,meta=rasterize(g,h)
  assert meta['central_clear_fraction'] >= .75-1e-12, '75% central clear-opening constraint failed'
  assert meta['total_line_of_sight_fraction'] >= .75-1e-12, '75% straight-through view constraint failed'
  if all(any(r['frequency_hz']==f and r['spacing_m']==h for r in records) for f in fs):continue
  mesh=PreparedMesh(s,np.arange(s.shape[1]+1)*dx,np.arange(s.shape[0]+1)*dz,boundary='periodic')
  for f in fs:
   if any(r['frequency_hz']==f and r['spacing_m']==h for r in records):continue
   r=mesh.solve(frequency=f)
   records.append(dict(frequency_hz=float(f),spacing_m=h,T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual))
   cache.write_text(json.dumps(records,indent=2));print(f'h={h:g} f={f:g} T={r.transmission:.7f}',flush=True)
 T=np.array([next(r['T'] for r in records if r['frequency_hz']==f and r['spacing_m']==.001) for f in freq]);weighted=power*gain;after=weighted*T
 with (OUT/'spectrum.csv').open('w') as file:
  w=csv.writer(file);w.writerow(['frequency_hz','measured_power','ear_power_gain','weighted_before','T_1mm_screening','weighted_after_1mm_screening'])
  w.writerows(zip(freq,power,gain,weighted,T,after))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.colors import ListedColormap
 fig,axs=plt.subplots(3,1,figsize=(12,11),layout='constrained',gridspec_kw={'height_ratios':[1,1.4,.85]})
 ax=axs[0];ax.semilogx(freq,T,'o-',color='#386cb0',label='1 mm mesh: computed at supplied band centres',markersize=4)
 fine=[r for r in records if r['spacing_m']==.0005]
 ax.scatter([r['frequency_hz'] for r in fine],[r['T'] for r in fine],marker='x',s=65,color='#ca4a22',label='0.5 mm mesh checks',zorder=5)
 ax.set(ylabel='Power transmission T',ylim=(-.03,1.06),title='Transmission of the earlier 75%-clear optimised candidate — UNVALIDATED')
 ax.legend(fontsize=9);ax.grid(alpha=.2)
 ax=axs[1]
 ax.semilogx(freq,power,'--',color='#949ba3',label='Measured band power')
 ax.semilogx(freq,weighted,'o-',color='#c87926',label='Measured × ear gain: before wall',markersize=4)
 ax.semilogx(freq,after,'o-',color='#177e80',label='Measured × ear gain × T: after candidate (screening)',markersize=4)
 ax.scatter([r['frequency_hz'] for r in fine],[weighted[np.where(freq==r['frequency_hz'])[0][0]]*r['T'] for r in fine],marker='x',s=65,color='#ca4a22',label='After candidate: 0.5 mm checks',zorder=5)
 ax.set(ylabel='Normalised band power (same scale)',xlabel='Frequency (Hz)',ylim=(0,1.08),title='Your weighted spectrum before and after the modelled transmission')
 ax.legend(fontsize=9);ax.grid(alpha=.2)
 for ax in axs[:2]:
  ax.axvspan(800,1250,color='#ebc96a',alpha=.14);ax.set_xlim(25,20000)
  ax.set_xticks([25,63,125,250,500,1000,2000,4000,8000,20000],['25','63','125','250','500','1k','2k','4k','8k','20k'])
 s,dx,dz,meta=rasterize(g,.001);ax=axs[2]
 ax.pcolormesh((np.arange(s.shape[0]+1)*dz-.024)*1000,np.arange(s.shape[1]+1)*dx*1000,s.T,cmap=ListedColormap(['#f0f5f6','#343b43']),shading='flat')
 ax.set_aspect('equal');ax.set(xlabel='Depth along sound propagation (mm) →',ylabel='Pitch (mm)',title='Geometry used: 200 mm pitch · 150 mm central clear opening · 240 mm core + 68 mm guides each side')
 fig.supxlabel('Screening only: rigid, lossless 2D periodic model at normal incidence; validation remains incomplete.\nLines join band-centre samples, not a resolved resonance sweep or band-averaged transmission. No separate renormalisation after attenuation.',fontsize=10)
 fig.savefig(OUT/'weighted_transmission.png',dpi=180);fig.savefig(OUT/'weighted_transmission.svg');plt.close(fig)
 ratio=float(after.sum()/weighted.sum());share=float(weighted[(freq>=800)&(freq<=1250)].sum()/weighted.sum())
 report=dict(opening_constraint=dict(minimum_clear_fraction=.75,central_clear_fraction=meta['central_clear_fraction'],total_straight_through_fraction=meta['total_line_of_sight_fraction'],scope='Repeating 2D cell; whole-wall frames and posts must also be counted in final design'),geometry=previous['geometry'],status=previous['candidate_status'],weighted_bandcentre_sum_before=float(weighted.sum()),weighted_bandcentre_sum_after_screening=float(after.sum()),screening_remaining_fraction=ratio,screening_reduction_db=float(-10*np.log10(ratio)),share_800_to_1250=share,maximum_energy_error=max(r['energy_error'] for r in records),method='Band-centre power transmission multiplied by supplied power and supplied A-weighting gain; no bandwidth factor. 1 mm periodic FV sweep, 0.5 mm checks.',limitations=['Not band-integrated: narrow resonances may be missed','No new optimisation; saved 75%-clear candidate, not scaled Figure 3 duct','Solver and geometry validation incomplete; no outdoor noise-reduction claim','Two cells across narrowest slit in 1 mm sweep; high-frequency and resonant values require convergence checks'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
 (OUT/'README.md').write_text('''# Yard spectrum screening

Run `.venv/bin/python -m acoustic_wall.yard_spectrum`. Saved solves resume automatically; remove the task-specific cache if geometry or solver changes.

Geometry is the earlier phase_guided_final candidate (200 mm pitch, 150 mm central clear channel, total depth 376 mm, side slits 2 and 3 mm). This is not the most recent scaled-paper 30.9%-clear duct. No new optimisation has occurred.

The supplied measured power and ear power gain are multiplied without intermediate rounding. After-wall weighted power = measured power × ear gain × total propagating power transmission. T is already a power ratio: it is not squared again. Before and after use the same normalisation.

The 30 band-centre solves use a 1 mm Cartesian mesh and rigid, lossless, normal-incidence periodic scattering with outgoing modal ports; 0.5 mm checks at 800, 1000 and 1250 Hz are overlaid. Existing numerical validation does not establish physical accuracy or convergence for this geometry. Joining samples does not resolve narrow resonances. Treat summed reduction as an illustrative band-centre estimate only, not a measured A-weighted reduction or guaranteed wall performance. A-weighting is a standard frequency weighting, not a complete loudness model. Band powers are summed without bandwidth factors; if these data are spectral densities instead, integration must be revised.

See spectrum.csv, solves.json and report.json for all plotted values and assumptions.
''')
if __name__=='__main__':main()
