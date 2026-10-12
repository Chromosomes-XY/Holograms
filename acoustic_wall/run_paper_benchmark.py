"""Run published fixed geometries; cache every case before making comparisons."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .paper_benchmark import CASES,ROOT,geometry,isolated_response
from .benchmark_mesh import PreparedMesh


def main():
    p=argparse.ArgumentParser();p.add_argument('--step-mm',type=float,default=.4)
    p.add_argument('--frequencies',default='1700:2300:20');p.add_argument('--boundary',default='rigid')
    p.add_argument('--convention',default='inclusive');p.add_argument('--labels',default='.4,.47,.53,.6,.67,.73')
    p.add_argument('--corner-mm',type=float);p.add_argument('--layout',default='centred');p.add_argument('--isolated',action='store_true');a=p.parse_args()
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    h=a.step_mm/1000;start=time.monotonic();rows=[]
    if ':' in a.frequencies:
        lo,hi,df=map(float,a.frequencies.split(':'));frequencies=np.arange(lo,hi+df/2,df)
    else:frequencies=np.array(list(map(float,a.frequencies.split(','))))
    for case in CASES:
        if case['label'] not in list(map(float,a.labels.split(','))):continue
        tag=f'o{case["label"]:.2f}_{a.convention}_{a.layout}_{a.boundary}_h{a.step_mm:g}'
        if a.corner_mm:tag+=f'_corner{a.corner_mm:g}'
        if a.isolated:
            path=out/(tag+'_isolated.json')
            if not path.exists():path.write_text(json.dumps(dict(case=case,step_m=h,convention=a.convention,layout=a.layout,response=isolated_response(case,h,a.convention,a.layout)),indent=2))
            print(path.name,flush=True);continue
        solid,xe,ze,meta=geometry(case,h,a.convention,a.layout,corner_step=a.corner_mm/1000 if a.corner_mm else None)
        model=None
        for f in frequencies:
            path=out/(tag+f'_f{f:g}.json');field_path=out/(tag+'_field.npz')
            if path.exists():continue
            if model is None:model=PreparedMesh(solid,xe,ze,a.boundary)
            r=model.solve(f)
            if r.balance_error>1e-7 or r.residual>1e-7:raise RuntimeError('Numerical conservation/residual failure')
            data=dict(meta,frequency_hz=float(f),boundary=a.boundary,T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual)
            path.write_text(json.dumps(data,indent=2))
            if f==2000:
                distance=np.linspace(0,.3,400)
                np.savez_compressed(field_path,pressure=r.pressure,solid=solid,xedges=xe,zedges=ze,
                                    before=r.extend(distance,True),after=r.extend(distance),distance=distance)
            rows.append(data)
        print(f'{tag}: {len(frequencies)} frequencies; elapsed {time.monotonic()-start:.1f}s',flush=True)
    print('finished',flush=True)

if __name__=='__main__':main()
