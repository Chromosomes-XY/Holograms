"""Bounded extra search: independently translate each rib train axially."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from .geometry import Geometry
from .phase_design import evaluate_phase
from .phase_optimise import objective_value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    g=Geometry(**json.loads(a.base.read_text())['geometry'])
    history=[];cache={};best=[float('inf'),None]
    def objective(v):
        key=tuple(int(round(x*1000)) for x in v)
        if key in cache:
            return cache[key]
        candidate=replace(g,offset_left=key[0]/1000,offset_right=key[1]/1000)
        d,m=evaluate_phase(candidate,.001)
        value=objective_value(d)
        item=dict(geometry=asdict(candidate),score=value,diagnostics=d,mesh=m)
        cache[key]=value;history.append(item)
        with (a.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(item)+'\n')
        if value<best[0]:
            best[:]=[value,item]
            (a.out/'best_so_far.json').write_text(json.dumps(item,indent=2))
        if len(history)%8==0 or value==best[0]:
            print(f'offset trial {len(history)}: best={best[0]:.4f}, T={d["transmission"]:.4f}, '
                  f'phase={np.round(d["regional_phase_deg"],1)}',flush=True)
        return value
    limit=min(.05,g.guide_length-.014)
    differential_evolution(objective,[(-limit,limit)]*2,maxiter=4,popsize=8,
                           seed=27,x0=[0,0],polish=False,tol=0,atol=0)
    (a.out/'report.json').write_text(json.dumps(best[1],indent=2))
    (a.out/'shortlist.json').write_text(json.dumps(sorted(history,key=lambda r:r['score'])[:8],indent=2))
    print(json.dumps(dict(geometry=best[1]['geometry'],score=best[0]),indent=2))


if __name__=='__main__':
    main()
