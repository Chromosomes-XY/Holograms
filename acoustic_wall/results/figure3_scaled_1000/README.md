# 1,000 Hz fixed-geometry solve — unvalidated

The latest Figure 3 reconstruction was scaled by two. The two solvers disagree and refinement does not establish convergence. Do not use either low-transmission result as a performance prediction.

Dimensions: core 171.5 mm, guides 68.6 mm each, total depth 308.7 mm, computational transverse width 162 mm, three clear channels 50 mm each, side slits 6.5 and 4 mm, axial ribs 10 mm, walls 4 mm. Equal rib pitch remains inferred. Central clear-view fraction including walls is 30.86%, not 75%.

See diagnostic_runs.json for all completed full-device runs; diagnostic_fields.npz contains the middle-grid FV cell and FEM nodal complex pressures and exact mesh/mask. pressure_1000_diagnostic.png displays both on one pressure colour scale with FEM incident phase aligned to the FV port centre. The model has rigid transverse boundaries and exact modal radiation at the axial ends; no absorption or thermoviscous losses.

Re-render: `.venv/bin/python -m acoustic_wall.plot_scaled_1000_diagnostic`.
The full fine-grid driver is solve_scaled_1000.py; its longer refinement run was stopped when disagreement appeared. Tests of independent_fem passed (three tests). scaling_check.json separately compares the identical dimensionless coarse mesh at 2,000 Hz and at 1,000 Hz after scaling by two. Passing scaling does not resolve the full-device solver discrepancy.
