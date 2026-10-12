# Phase-guided reoptimisation: results

Selected geometry: {'pitch': 0.2, 'core_depth': 0.24, 'guide_length': 0.068, 'central_fraction': 0.75, 'wall': 0.004, 'rib_thickness': 0.01, 'rib_count': 10, 'slit_left': 0.002, 'slit_right': 0.003, 'offset_left': 0.0, 'offset_right': 0.0}

| Grid spacing (mm) | Power transmission at 1 kHz | Transmission loss (dB) |
|---|---:|---:|
| 1.000 | 0.00006 | 41.87 |
| 0.500 | 0.00101 | 29.94 |
| 0.333 | 0.00227 | 26.43 |
| 0.250 | 0.00306 | 25.14 |

Target relative exit phases: [-157.5, 0.0, 157.5] degrees.
Measured relative exit-pressure phases: [-174.7543246252769, -7.526945820486311e-16, -176.3765495293654] degrees.
Near-field evanescent pressure-squared fraction: 0.9985.
This is not a transmitted-power fraction.

Screening checks: {'transmission_below_point_one': True, 'phase_within_15_degrees': False, 'majority_near_field_pressure_evanescent': True, 'significant_first_evanescent_orders': True, 'mesh_transmission_change_below_point_zero_two': True, 'mesh_tl_change_below_one_db': False, 'mesh_phase_change_below_10_degrees': True}

The cell is not established as a bound travelling surface eigenmode. In
particular, low transmission alone does not establish the intended phase gradient.
See report.json for channel decompositions, all outgoing orders, angular and
frequency checks, dimensional perturbations and mesh comparisons. The ideal
amplitude target is a prescribed-pressure aperture surrogate, not a reproduction
of the paper's COMSOL port optimisation. This is a rigid lossless 2D periodic
calculation, not an outdoor noise reduction guarantee.
