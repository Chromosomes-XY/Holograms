"""Straight, open-sided rectangular notch controls on the same 5 mm grid."""
import json
import numpy as np
from .manufacturable_search import OUT,SEEDS,propose,evaluate,geometry

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True);selected=[]
    for count,seed in SEEDS.items():
        path=OUT/f'notch_history_{count}.json';history=json.loads(path.read_text()) if path.exists() else [evaluate(seed,notches=True)]
        best=min([q for q in history if not q['pruned']],key=lambda q:q['J']);rng=np.random.default_rng(820+count)
        for step in range(len(history)-1,20):
            v=propose(best['v'],rng,step)
            # Open notches have no independent neck/mouth parameter.
            for i in range(count):v[3*i+2]=seed[3*i+2]
            q=evaluate(v,cutoff=best['J'],notches=True);history.append(q)
            if not q['pruned'] and q['J']<best['J']:best=q;print('Notch improvement',count,step+1,best['J'],flush=True)
            path.write_text(json.dumps(history))
        finalists=[]
        for q in sorted([q for q in history if not q['pruned']],key=lambda q:q['J'])[:2]:
            finalists.append(evaluate(q['v'],n=9,h=.0025,notches=True,dominant_n=33))
        best=min(finalists,key=lambda q:q['J']);best['cavity_count']=count;selected.append(best)
        print('Notch selected',count,best['J'],best['depth_mm'],flush=True)
    (OUT/'notch_report.json').write_text(json.dumps(dict(selected=selected,proposal_count_per_family=20,
        geometry='Each rectangular recess opens over its entire axial length; no narrow mouth, no lip, no hidden undercut. Cut penetration is width+5 mm. Source parameter mouth is unused.',
        limitations='Limited local search; ideal square corners. Cutter radius, joint gaps and material compliance not modelled.'),indent=2))

if __name__=='__main__':main()
