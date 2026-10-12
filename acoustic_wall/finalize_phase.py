"""Combine completed refinement and independent quarter-mm check; no new solve."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .geometry import Geometry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('refined',type=Path)
    p.add_argument('inspection',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.refined.read_text());extra=json.loads(a.inspection.read_text())
    if asdict(Geometry(**data['geometry']))!=asdict(Geometry(**extra['geometry'])):
        raise ValueError('Geometry mismatch: cannot combine these checks')
    old=data['refinement'][-1]['diagnostics'];fine=extra['diagnostics']
    data['refinement'].append(dict(spacing_m=extra['mesh']['dz_m'],mesh=extra['mesh'],diagnostics=fine))
    data['final_mesh_diagnostics']=fine
    data['directional_channels']=extra['directional_channels']
    data['mesh_transmission_change']=abs(fine['transmission']-old['transmission'])
    data['mesh_tl_change_db']=abs(fine['tl_db']-old['tl_db'])
    data['mesh_phase_change_deg']=float(np.max(abs(np.angle(np.exp(1j*np.deg2rad(
        np.array(fine['regional_phase_deg'])-np.array(old['regional_phase_deg']))),deg=True))))
    gates=dict(transmission_below_point_one=fine['transmission']<.1,
               phase_within_15_degrees=bool(max(abs(np.array(fine['regional_phase_errors_deg'])[[0,2]]))<15),
               majority_near_field_pressure_evanescent=fine['near_evanescent_pressure_squared_fraction']>.5,
               significant_first_evanescent_orders=fine['first_order_evanescent_rms']>.1,
               mesh_transmission_change_below_point_zero_two=data['mesh_transmission_change']<.02,
               mesh_tl_change_below_one_db=data['mesh_tl_change_db']<1.,
               mesh_phase_change_below_10_degrees=data['mesh_phase_change_deg']<10.)
    data['mechanism_checks']=gates;data['all_screening_checks_pass']=all(gates.values())
    data['candidate_status']='screening candidate; all checks passed' if all(gates.values()) else 'incomplete mechanism/convergence validation'
    data['geometry']=extra['geometry']
    data['normal_incidence_final_mesh_m']=extra['mesh']['dz_m']
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'report.json').write_text(json.dumps(data,indent=2))
    table='\n'.join(f'| {r["spacing_m"]*1000:.3f} | {r["diagnostics"]["transmission"]:.5f} | {r["diagnostics"]["tl_db"]:.2f} |'
                    for r in data['refinement'])
    text=f'''# Phase-guided reoptimisation: results

Selected geometry: {data['geometry']}

| Grid spacing (mm) | Power transmission at 1 kHz | Transmission loss (dB) |
|---|---:|---:|
{table}

Target relative exit phases: {fine['target_phase_deg']} degrees.
Measured relative exit-pressure phases: {fine['regional_phase_deg']} degrees.
Near-field evanescent pressure-squared fraction: {fine['near_evanescent_pressure_squared_fraction']:.4f}.
This is not a transmitted-power fraction.

Screening checks: {gates}

The cell is not established as a bound travelling surface eigenmode. In
particular, low transmission alone does not establish the intended phase gradient.
See report.json for channel decompositions, all outgoing orders, angular and
frequency checks, dimensional perturbations and mesh comparisons. The ideal
amplitude target is a prescribed-pressure aperture surrogate, not a reproduction
of the paper's COMSOL port optimisation. This is a rigid lossless 2D periodic
calculation, not an outdoor noise reduction guarantee.
'''
    (a.out/'RESULTS.md').write_text(text)
    print(text)


if __name__=='__main__':
    main()
