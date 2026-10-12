"""Staged channel characterisation and phase/amplitude-guided geometry search."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import differential_evolution
from .geometry import Geometry
from .phase_design import isolated_channels,evaluate_phase,phase_targets,ideal_target


def objective_value(d):
    phase=np.deg2rad(np.array(d['regional_phase_errors_deg'])[[0,2]])
    # Explicit relative-phase term prevents the small central target amplitude
    # from making a wrong central phase appear acceptable in a vector norm.
    phase_loss=float(np.mean((1-np.cos(phase))/2)) if d['phase_valid'] else 1.
    return .55*d['transmission']+.25*phase_loss+.20*d['target_match_error']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=6)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--seed',type=int,default=26)
    a=p.parse_args()
    if a.population<8 or a.iterations<0:
        p.error('population >=8 and iterations >=0 required')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Use a fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    g=Geometry()
    old=json.loads((Path(__file__).parent/'results/target_1000_search/report.json').read_text())
    previous=Geometry(**old['geometry'])
    library=[]; seeds=[]
    # Characterise the entire physical side channel at 1 mm resolution. This is
    # a coarse response library; fine-mesh validation follows full-cell search.
    target=phase_targets(.75)
    for core_mm in (140,160,180,200,220,240):
        responses=[]
        for slit_mm in range(2,17):
            candidate=replace(g,core_depth=core_mm/1000,slit_left=slit_mm/1000,
                              slit_right=slit_mm/1000)
            t,records=isolated_channels(candidate,.001)
            relative=t[0]/t[1]
            item=dict(core_mm=core_mm,slit_mm=slit_mm,amplitude=float(abs(t[0])),
                      relative_phase_deg=float(np.angle(relative,deg=True)),channels=records)
            library.append(item);responses.append((slit_mm,relative))
        choices=[]
        for desired in (target[0],target[2]):
            # Phase first, but disfavor almost completely closed/transmitting-zero channels.
            ranked=sorted(responses,key=lambda pair:abs(np.angle(pair[1]*np.exp(-1j*desired)))
                          +.15*(1-abs(pair[1])))
            choices.append(ranked[0][0]/1000)
        seeds.append([choices[0],choices[1],core_mm/1000,.068])
        print(f'channel library core={core_mm} mm: seed slits={choices}',flush=True)
    library_report=dict(records=library,phase_targets_deg=np.rad2deg(target).tolist(),
                        ideal_nominal_amplitudes=abs(ideal_target()).tolist(),
                        ideal_model='prescribed pressure aperture; side amplitudes fixed to 1; not passive device',
                        paper_port_replication=False)
    (a.out/'channel_library.json').write_text(json.dumps(library_report,indent=2))
    history=[];cache={}
    best=[float('inf'),None]
    def objective(v):
        key=tuple(int(round(x*1000)) for x in v)
        if key in cache:
            return cache[key]
        candidate=replace(g,slit_left=key[0]/1000,slit_right=key[1]/1000,
                          core_depth=key[2]/1000,guide_length=key[3]/1000)
        try:
            d,m=evaluate_phase(candidate,.001)
        except ValueError:
            cache[key]=1e3;return 1e3
        value=objective_value(d)
        item=dict(geometry=asdict(candidate),score=value,diagnostics=d,mesh=m)
        history.append(item);cache[key]=value
        with (a.out/'history.jsonl').open('a') as f:
            f.write(json.dumps(item)+'\n')
        if value<best[0]:
            best[:]=[value,item]
            (a.out/'best_so_far.json').write_text(json.dumps(item,indent=2))
        if len(history)%5==0 or value==best[0]:
            print(f'evaluation {len(history)} best={best[0]:.4f}; current T={d["transmission"]:.4f}, '
                  f'phase={np.round(d["regional_phase_deg"],1)}',flush=True)
        return value
    bounds=np.array([[.002,.016],[.002,.016],[.130,.260],[.020,.120]])
    rng=np.random.default_rng(a.seed)
    population=rng.uniform(bounds[:,0],bounds[:,1],size=(a.population,4))
    seeds.append([previous.slit_left,previous.slit_right,previous.core_depth,previous.guide_length])
    population[:len(seeds)]=seeds
    differential_evolution(objective,bounds,init=population,maxiter=a.iterations,
                           seed=a.seed,polish=False,workers=1,tol=0,atol=0)
    report=dict(best[1],candidate_status='1 mm screening optimum; awaiting finer-grid selection',
                objective='.55 T + .25 relative phase loss + .20 complex amplitude/phase pattern loss',
                frequency_hz=1000,angle_deg=0,unique_evaluations=len(history),
                elapsed_s=time.monotonic()-start,seed=a.seed,
                limitations='2D rigid lossless periodic; ideal target is an aperture surrogate, not paper port replication')
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    shortlist=sorted(history,key=lambda x:x['score'])[:8]
    (a.out/'shortlist.json').write_text(json.dumps(shortlist,indent=2))
    print(json.dumps({k:report[k] for k in ['geometry','score','unique_evaluations','elapsed_s']},indent=2))


if __name__=='__main__':
    main()
