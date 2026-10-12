# Finite 9.9 m wall: horizontal 2D screening simulation

33 copies of the selected 300 mm cell form a 9.9 m wall, 373.4 mm deep, with exactly 75% frontal clear opening. The cavities are explicitly meshed. There are no periodic boundaries or effective transmission screens in this model.

The source is at x=0, z=-3 m; the wall front is z=0. This is a 2D point source, equivalent to an infinite vertical line source in 3D. The intended 1.8 m wall height is recorded but cannot be represented in a horizontal 2D section. There is no over-top diffraction, ground reflection, structural vibration, flow or thermoviscous loss.

## Numerical method

Solve the total-field finite-volume Helmholtz equation in air with rigid/zero-normal-velocity solid faces. Nonuniform rectangular cells preserve all cavity interfaces, including the 3 mm mouth walls. Both wall and no-wall cases use identical grids and bilinearly distributed monopole forcing. Exterior stretched-coordinate PML absorbs outgoing waves, followed by a homogeneous Dirichlet outer boundary. The physical domain extends x=±6.75 m and z=-3.75..4.75 m; the usual PML is 0.75 m thick.

The coarse mesh has 222,276 air unknowns, up to 20 mm cells near the wall and 40 mm away. Because its 1 kHz result was sensitive to resolution, the final five dominant bands (630, 800, 1000, 1250 and 1600 Hz) use a finer 747,136-unknown mesh with 10/20 mm maximum steps. Fine-grid power is sampled onto the plotting grid using nearest-cell sampling; the 1 kHz pressure snapshot uses the fine grid directly. These are screening meshes, not a spatial convergence claim.

The original analytical-incident/scattered-field trial is archived in `discarded_analytic_incident/` and is **not** used in the final results. Combining an exact incident wave with a dispersive coarse scattered wave caused misleading interference. The total-field formulation avoids that mismatch.

## Spectrum and plots

Every supplied band through 1600 Hz is solved at its centre frequency: 94.8503% of the supplied A-weighted input power. The rest of the spectrum is added as an analytical, unattenuated background for the approximate whole-spectrum maps. This is an assumption, not a claim that those frequencies physically pass unchanged or a rigorous local error bound.

Band powers combine incoherently: sum P_i × ear_gain_i × |p_i|². Do not sum pressure amplitudes across unrelated noise frequencies. The supplied band powers are not multiplied by bandwidth again. There is no within-band averaging, unlike the preceding unit-cell optimisation; narrow resonances can make these centre-frequency results differ substantially.

The source forcing is calibrated to the analytic 2D Hankel field with unit pressure magnitude at 1 m per unit input power. Absolute measured SPL was not supplied, so colour levels are relative and are not calibrated dBA. Insertion loss is 10 log10(no-wall mean-square pressure / wall mean-square pressure) at each point. Positive values are quieter; negative values show local enhancement. It is a pressure comparison, not a transmitted-power flux measurement.

`finite_wall_maps.png` contains the no-wall reference, wall weighted level, insertion-loss map, and a single-phase 1000 Hz pressure snapshot. `geometry_detail.png` shows four explicit cells. `broadband.npz` contains summed powers, and per-frequency NPZ files retain complex fields. `report.json` and `checks.json` record parameters and numerical sensitivity.

## Reproduce

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.finite_array_2d
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.refine_finite_array
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.check_finite_array
```

The last command generates `pml_check_1000.npz` if absent, using `FiniteArray(pml=1.5)` with wall and empty-domain solves at 1000 Hz. Doubling PML thickness changed the selected receiver-region average by less than 0.0001 dB on the coarse grid; this tests boundary reflection only, not spatial accuracy. Cached field files are model-specific and must be invalidated if geometry, source or solver changes.
