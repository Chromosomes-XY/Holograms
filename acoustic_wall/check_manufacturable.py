"""Spatial and angular screening of the best discrete construction options."""
import json
import numpy as np
from .manufacturable_search import OUT,geometry,PreparedMesh,evaluate,FC,EDGES,WEIGHT

def check(q,notches):
    path=OUT/('notch_checks.json' if notches else 'pocket_checks.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old['v']==q['v']:return old
    rows=[];nodes,w=np.polynomial.legendre.leggauss(9)
    for h in [.0025,.00125]:
        s,x,z=geometry(q['v'],h,notches=notches);mesh=PreparedMesh(s,x,z,boundary='periodic')
        for f in [800,1000,1250]:
            i=int(np.flatnonzero(FC==f)[0]);lo,hi=EDGES[i:i+2];fs=(lo+hi)/2+(hi-lo)*nodes/2
            ts=[mesh.solve(float(f)).transmission for f in fs]
            rows.append(dict(h=h,f=f,T=float(w@ts/2),weight=float(WEIGHT[i])))
        print('Mesh checked',notches,h,flush=True)
    angles=[]
    for angle in [0,-30,30,-60,60]:
        angles.append(evaluate(q['v'],n=5,h=.0025,angle=angle,notches=notches))
        print('Angle checked',notches,angle,angles[-1]['J'],flush=True)
    r=dict(v=q['v'],notches=notches,spatial_checks=rows,angle_checks=angles,
           limitations='Angle checks use five samples per band and are screening estimates. Normal final scores use denser quadrature. No directional energy weights assumed; search was normal-incidence only.')
    path.write_text(json.dumps(r,indent=2));return r

def main():
    pocket=json.loads((OUT/'report.json').read_text());notch=json.loads((OUT/'notch_report.json').read_text())
    check(min(pocket['selected'],key=lambda q:q['J']),False)
    check(min(notch['selected'],key=lambda q:q['J']),True)

if __name__=='__main__':main()
