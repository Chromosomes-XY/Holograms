"""Independent extra-mesh check, including directional channel decomposition."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import Geometry,rasterize
from .solver import solve
from .phase_design import diagnostics,directional_channels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--spacing-mm',type=float,default=.25)
    a=p.parse_args()
    g=Geometry(**json.loads(a.report.read_text())['geometry'])
    mask,dx,dz,mesh=rasterize(g,a.spacing_mm/1000)
    result=solve(mask,dx,dz)
    d=diagnostics(g,mask,result)
    channels=directional_channels(g,mask,result)
    if result.balance_error>1e-7 or result.residual>1e-7:
        raise RuntimeError('Solve failed verification')
    a.out.mkdir(parents=True,exist_ok=True)
    from dataclasses import asdict
    report=dict(geometry=asdict(g),mesh=mesh,diagnostics=d,directional_channels=channels)
    (a.out/'inspection.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(a.out/'field.npz',pressure=result.pressure,solid=mask,dx=dx,dz=dz,
                        kx=result.kx,qz=result.qz,outgoing_modes=result.outgoing_modes)
    print(json.dumps(dict(transmission=d['transmission'],tl_db=d['tl_db'],
                         phase=d['regional_phase_deg'],evanescent=d['near_evanescent_pressure_squared_fraction'],
                         imbalance=d['first_order_directional_imbalance'],channels=channels),indent=2))


if __name__=='__main__':
    main()
