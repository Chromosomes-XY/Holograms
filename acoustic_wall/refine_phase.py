"""Fine-grid reranking, local refinement and mechanism checks for phase search."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import hashlib
import numpy as np
from .geometry import Geometry
from .phase_design import evaluate_phase,isolated_channels
from .phase_optimise import objective_value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('search',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shortlist',type=int,default=5)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.out.exists() and any(a.out.iterdir()) and not a.resume:
        p.error('Use a fresh output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();history=[];cache={}
    if a.resume and (a.out/'fine_history.jsonl').exists():
        for line in (a.out/'fine_history.jsonl').read_text().splitlines():
            item=json.loads(line)
            cache[tuple(asdict(Geometry(**item['geometry'])).values())]=item
            history.append(item)
    def checked(g,h,frequency=1000.,angle=0.):
        params=dict(geometry=asdict(g),spacing=h,frequency=frequency,angle=angle)
        key=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]
        folder=a.out/'validation_cache';folder.mkdir(exist_ok=True)
        path=folder/(key+'.json')
        if path.exists():
            saved=json.loads(path.read_text());return saved['diagnostics'],saved['mesh']
        d,m=evaluate_phase(g,h,frequency,angle)
        path.write_text(json.dumps(dict(parameters=params,diagnostics=d,mesh=m),indent=2))
        return d,m
    def assess(g):
        key=tuple(asdict(g).values())
        if key in cache:
            return cache[key]
        d,m=checked(g,.0005)
        item=dict(geometry=asdict(g),score=objective_value(d),diagnostics=d,mesh=m)
        cache[key]=item;history.append(item)
        with (a.out/'fine_history.jsonl').open('a') as f:
            f.write(json.dumps(item)+'\n')
        print(f'fine solve {len(history)}: T={d["transmission"]:.4f}, '
              f'phase={np.round(d["regional_phase_deg"],1)}, score={item["score"]:.4f}',flush=True)
        return item
    shortlist=json.loads((a.search/'shortlist.json').read_text())[:a.shortlist]
    best=min((assess(Geometry(**item['geometry'])) for item in shortlist),key=lambda x:x['score'])
    # Two deterministic coordinate sweeps; all selection is on the 0.5 mm mesh.
    for sweep in range(2):
        before=best['score']
        for name,step in [('slit_left',.0005),('slit_right',.0005),
                          ('core_depth',.002),('guide_length',.002)]:
            g=Geometry(**best['geometry'])
            for delta in (-step,step):
                candidate=replace(g,**{name:getattr(g,name)+delta})
                try:
                    item=assess(candidate)
                except ValueError:
                    continue
                if item['score']<best['score']:
                    best=item
        (a.out/'best_so_far.json').write_text(json.dumps(best,indent=2))
        if best['score']>=before-1e-8:
            break
    g=Geometry(**best['geometry'])
    # Refine without moving nominal geometry. Grid errors include raster changes.
    refinement=[]
    for h in (.001,.0005,1/3000):
        d,m=checked(g,h)
        refinement.append(dict(spacing_m=h,diagnostics=d,mesh=m))
        print(f'validation mesh {h*1000:.3f} mm T={d["transmission"]:.5f}',flush=True)
    # Short-band and angular checks use the optimisation's 0.5 mm mesh.
    checks=[]
    for f in (950.,975.,1000.,1025.,1050.):
        for angle in (0.,30.,-30.):
            d,_=checked(g,.0005,f,angle)
            checks.append(dict(frequency_hz=f,angle_deg=angle,diagnostics=d))
        print(f'checked frequency {f:g} Hz',flush=True)
    _,channel_records=isolated_channels(g,.0005)
    perturb=[]
    for key in ('slit_left','slit_right','core_depth','guide_length'):
        for delta in (-.0005,.0005):
            try:
                d,_=checked(replace(g,**{key:getattr(g,key)+delta}),.0005)
            except ValueError:
                continue
            perturb.append(dict(parameter=key,delta_m=delta,transmission=d['transmission'],
                                tl_db=d['tl_db'],phase_errors_deg=d['regional_phase_errors_deg']))
    fine=refinement[-1]['diagnostics'];middle=refinement[-2]['diagnostics']
    transmission_change=abs(fine['transmission']-middle['transmission'])
    phase_change=float(np.max(abs(np.angle(np.exp(1j*np.deg2rad(
        np.array(fine['regional_phase_deg'])-np.array(middle['regional_phase_deg']))),deg=True))))
    gates=dict(transmission_below_point_one=fine['transmission']<.1,
               phase_within_15_degrees=max(abs(np.array(fine['regional_phase_errors_deg'])[[0,2]]))<15,
               majority_near_field_pressure_evanescent=fine['near_evanescent_pressure_squared_fraction']>.5,
               significant_first_evanescent_orders=fine['first_order_evanescent_rms']>.1,
               mesh_transmission_change_below_point_zero_two=transmission_change<.02,
               mesh_phase_change_below_10_degrees=phase_change<10)
    gates={key:bool(value) for key,value in gates.items()}
    report=dict(best,final_mesh_diagnostics=fine,refinement=refinement,checks=checks,
                isolated_channels=channel_records,perturbations=perturb,
                mechanism_checks=gates,all_screening_checks_pass=all(gates.values()),
                mesh_transmission_change=transmission_change,mesh_phase_change_deg=phase_change,
                elapsed_s=time.monotonic()-start,fine_geometry_evaluations=len(history),
                bound_surface_eigenmode_proven=False,
                limitations='Lossless 2D periodic screening. Pressure evanescence is not an eigenmode proof. '
                            'Ideal target is an aperture surrogate, not the authors port-model reproduction.')
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(geometry=asdict(g),checks=gates,tl_db=fine['tl_db'],
                         phase_deg=fine['regional_phase_deg'],elapsed_s=report['elapsed_s']),indent=2))


if __name__=='__main__':
    main()
