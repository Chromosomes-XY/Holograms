"""Resolve frequency quadrature in the dominant bands after initial selection."""
import copy
from .optimise_measured_noise import OUT, PreparedMesh, geometry, EDGES, FC, POWER, WEIGHT, plot
import numpy as np
import json

def main():
    path=OUT/'report.json';r=json.loads(path.read_text())
    if 'dominant_band_refinement' in r:return
    history=[]
    for key in ['baseline','selected']:
        q=r[key];s,x,z=geometry(q['v'],.0015);mesh=PreparedMesh(s,x,z,boundary='periodic')
        for n in [33,65]:
            previous=q['J'];nodes,w=np.polynomial.legendre.leggauss(n);w=w/2
            for f in [800,1000,1250]:
                i=int(np.flatnonzero(FC==f)[0]);lo,hi=EDGES[i:i+2];fs=(lo+hi)/2+(hi-lo)*nodes/2
                results=[mesh.solve(float(frequency)) for frequency in fs]
                ts=[v.transmission for v in results];t=float(w@ts)
                q['bands'][i].update(T=t,after=float(POWER[i]*t),frequencies=fs.tolist(),transmission=ts,
                                     errors=[v.balance_error for v in results],residuals=[v.residual for v in results])
            q['J']=float(sum(b['after'] for b in q['bands'])/POWER.sum());q['reduction_db']=float(-10*np.log10(q['J']))
            history.append(dict(geometry=key,n=n,J=q['J'],change=q['J']-previous))
            print(history[-1],flush=True)
            if abs(q['J']-previous)<.002:break
    r['dominant_band_refinement']=history
    r['integration_label']='17 points per band; 800, 1000 and 1250 Hz bands refined to 33–65 points'
    path.write_text(json.dumps(r,indent=2));plot(r)
    with (OUT/'README.md').open('a') as f:
        f.write('\nAdditional dominant-band frequency refinement: run `python -m acoustic_wall.refine_measured_bands`. The 800, 1000 and 1250 Hz bands receive 33, then up to 65 Gauss points, stopping when the change in total J is below 0.002. Other bands retain 17 points. See report.json for achieved changes; this is a convergence check, not a strict error bound.\n')

if __name__=='__main__':main()
