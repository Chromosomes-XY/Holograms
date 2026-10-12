"""Horizontal oblique-incidence sweep of the fixed 75%-open selected design."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import json,csv
from pathlib import Path
import numpy as np
from .expand_300 import geometry,OUT as DESIGN
from .bloch_check import solve_bloch
OUT=DESIGN/'angles_30_60'

def main():
 OUT.mkdir(parents=True,exist_ok=True);design=json.loads((DESIGN/'report.json').read_text());v=design['selected']['v'];cache=OUT/'runs.json';rows=json.loads(cache.read_text()) if cache.exists() else []
 # Reuse previously completed exact same geometry/grid/solver spot checks.
 if not rows:
  for q in design.get('angle_checks_1p5mm',[]):
   if q['angle_deg'] in [-30,0,30]:rows.append(dict(q,mesh_m=.0015))
 for h,angles,fs in [(.0015,[-60,-30,0,30,60],np.arange(700,1401,25)),(.00075,[-60,-30,30,60],[800,1000,1250])]:
  s,x,z=geometry(v,h)
  for angle in angles:
   for f in fs:
    if any(q['angle_deg']==angle and q['frequency_hz']==f and q['mesh_m']==h for q in rows):continue
    q=solve_bloch(s,x,z,float(f),angle);q['mesh_m']=h
    assert -1e-8<=q['T']<=1+1e-8 and q['energy_error']<1e-8
    rows.append(q);cache.write_text(json.dumps(rows,indent=2))
   print('Completed',h,angle,flush=True)
 # Independent invariants specifically at 60 degrees.
 s,x,z=geometry(v,.003);ze=np.linspace(-.02,.42,221)
 empty=solve_bloch(np.zeros((220,len(x)-1),bool),x,ze,1250,60);assert abs(empty['T']-1)<1e-8
 m=np.r_[v[3:6],v[:3],v[9:12],v[6:9],v[13],v[12],v[15],v[14]];sm,xm,zm=geometry(m,.003)
 a=solve_bloch(s,x,z,1000,60);b=solve_bloch(sm,xm,zm,1000,-60);assert abs(a['T']-b['T'])<1e-8
 fs=np.arange(700,1401,25);weights=np.interp(fs,[700,800,1000,1250,1400],[.311,.622,1,.5485,.137125]);weights[[0,-1]]*=.5;weights/=weights.sum()
 summaries=[]
 for angle in [-60,-30,0,30,60]:
  rs=sorted([q for q in rows if q['mesh_m']==.0015 and q['angle_deg']==angle and q['frequency_hz'] in fs],key=lambda q:q['frequency_hz'])
  summaries.append(dict(angle_deg=angle,weighted_mean_T=float(np.array([q['T'] for q in rs])@weights),points=[q for q in rs if q['frequency_hz'] in [800,1000,1250]]))
 report=dict(design=str(DESIGN/'report.json'),angle_definition='Horizontal incidence from the wall normal; both signs tested because cavity arrangement is asymmetric.',geometry=design['cavities'],clear_fraction=.75,pitch_mm=300,depth_mm=design['depth_mm'],summaries=summaries,fine_checks=[q for q in rows if q['mesh_m']==.00075],max_energy_error=max(q['energy_error'] for q in rows),empty_60deg=empty,mirror_60deg_error=abs(a['T']-b['T']),limits=['Fixed design; no angular reoptimisation','Normalised by incident power flux normal to wall; sums all transmitted propagating orders','1.5 mm full sweep, 0.75 mm spot checks; 25 Hz sampling may miss narrow resonances','Rigid lossless periodic 2D array; no screens, finite wall, ground or elevation-angle effects','Weighted mean uses the same interpolated spectral envelope at every angle; not total measured A-weighted noise reduction'])
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 with (OUT/'transmission.csv').open('w') as file:
  w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained',sharex=True)
 for ax,mag in zip(axs,[30,60]):
  for angle,color,ls in [(0,'#8e969d','-'),(mag,'#087f83','-'),(-mag,'#c47729','--')]:
   rr=sorted([q for q in rows if q['mesh_m']==.0015 and q['angle_deg']==angle],key=lambda q:q['frequency_hz']);ax.plot([q['frequency_hz'] for q in rr],[q['T'] for q in rr],label=f'{angle:+d}°' if angle else '0° (normal)',color=color,ls=ls)
   fine=[q for q in rows if q['mesh_m']==.00075 and q['angle_deg']==angle]
   if fine:ax.scatter([q['frequency_hz'] for q in fine],[q['T'] for q in fine],color=color,marker='x',s=45)
  ax.axvspan(800,1250,color='#edd48e',alpha=.12);ax.set(ylabel='Total transmitted power fraction T',ylim=(-.02,1.04),title=f'Incidence at ±{mag}° from the wall normal');ax.grid(alpha=.2);ax.legend()
 axs[-1].set_xlabel('Frequency (Hz)');fig.suptitle('Selected 300 mm / 75%-open design: sideways incidence',fontsize=17)
 fig.supxlabel('Lines: 1.5 mm mesh, 25 Hz sweep. Crosses: 0.75 mm checks. All outgoing diffraction orders included.\nHorizontal 2D, rigid lossless infinite periodic wall; no insect screens or finite-height effects. Not an outdoor noise rating.',fontsize=10)
 fig.savefig(OUT/'angular_transmission.png',dpi=180);plt.close(fig)
 (OUT/'README.md').write_text('''# Sideways incidence at 30 and 60 degrees

Run `.venv/bin/python -m acoustic_wall.check_angles_30_60`.

Uses the fixed selected expanded_300_75pct geometry: 300 mm period, 225 mm central clear opening, 360.2 mm depth. Angles are measured from the wall normal in the horizontal cross-section; both signs are checked. This does not address elevation-angle incidence.

Bloch-phase transverse boundaries and outgoing modal ports, with the incident transverse mode selected by its overlap with the oblique plane wave. Power is normalised by incident normal flux, including every propagating outgoing diffraction order. Sweep 700–1400 Hz every 25 Hz on the 1.5 mm interface-aligned mesh; 0.75 mm checks at 800, 1000 and 1250 Hz for ±30 and ±60 degrees. Additional empty-space and mirrored-geometry/sign-reversal checks at 60 degrees passed. Results are saved in report.json, runs.json and transmission.csv.

The weighted summary uses the same interpolated spectral envelope for each angle with trapezoidal sampling weights. It is a comparative screening metric, not full measured-band integration or an outdoor A-weighted noise rating. The model omits screens, losses, finite height/length, ground and supports. No reoptimisation occurred. The 75% frontal opening is not a claim of 75% view or airflow projected at oblique angles.
''')
 print('SUMMARY',[(q['angle_deg'],q['weighted_mean_T'],[(a['frequency_hz'],a['T']) for a in q['points']]) for q in summaries],flush=True)
if __name__=='__main__':main()
