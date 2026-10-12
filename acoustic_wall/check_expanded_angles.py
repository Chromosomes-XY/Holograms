"""Angular spot checks for the selected expanded candidate (not angle optimisation)."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,csv
from .expand_300 import OUT,geometry
from .bloch_check import solve_bloch

def main():
 path=OUT/'report.json';r=json.loads(path.read_text());v=r['selected']['v'];s,x,z=geometry(v,.0015);p=OUT/'angle_checks.json';rows=json.loads(p.read_text()) if p.exists() else []
 for angle in [-30,-15,0,15,30]:
  for f in [800,1000,1250]:
   if any(q['angle_deg']==angle and q['frequency_hz']==f for q in rows):continue
   q=solve_bloch(s,x,z,f,angle);assert q['energy_error']<1e-8 and 0<=q['T']<=1+1e-8
   rows.append(q);p.write_text(json.dumps(rows,indent=2));print(angle,f,q['T'],flush=True)
 r['angle_checks_1p5mm']=rows
 for q in rows:
  if q['angle_deg']==0:
   ref=next(a for a in r['selected']['responses'] if a['f']==q['frequency_hz']);assert abs(q['T']-ref['T'])<1e-8
 r['normal_incidence_crosscheck_passed']=True;path.write_text(json.dumps(r,indent=2))
 with (OUT/'geometry_mm.csv').open('w') as file:
  w=csv.DictWriter(file,fieldnames=list(r['cavities'][0]));w.writeheader();w.writerows(r['cavities'])
 with (OUT/'README.md').open('a') as file:file.write('\nAdditional spot checks: `python -m acoustic_wall.check_expanded_angles` evaluates −30°, −15°, 0°, +15°, +30° at 800, 1000 and 1250 Hz on the same 1.5 mm interface-aligned mesh. This uses Bloch seam phases and a weighted Hermitian modal-port decomposition, selects incidence by plane-wave overlap, and sums all outgoing propagating modes. The search itself was normal-incidence only. See bloch_verification.json, angle_checks.json and geometry_mm.csv.\n')
if __name__=='__main__':main()
