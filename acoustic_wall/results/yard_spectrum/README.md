# Yard spectrum screening

Run `.venv/bin/python -m acoustic_wall.yard_spectrum`. Saved solves resume automatically; remove the task-specific cache if geometry or solver changes.

Geometry is the earlier phase_guided_final candidate (200 mm pitch, 150 mm central clear channel, total depth 376 mm, side slits 2 and 3 mm). This is not the most recent scaled-paper 30.9%-clear duct. No new optimisation has occurred.

The supplied measured power and ear power gain are multiplied without intermediate rounding. After-wall weighted power = measured power × ear gain × total propagating power transmission. T is already a power ratio: it is not squared again. Before and after use the same normalisation.

The 30 band-centre solves use a 1 mm Cartesian mesh and rigid, lossless, normal-incidence periodic scattering with outgoing modal ports; 0.5 mm checks at 800, 1000 and 1250 Hz are overlaid. Existing numerical validation does not establish physical accuracy or convergence for this geometry. Joining samples does not resolve narrow resonances. Treat summed reduction as an illustrative band-centre estimate only, not a measured A-weighted reduction or guaranteed wall performance. A-weighting is a standard frequency weighting, not a complete loudness model. Band powers are summed without bandwidth factors; if these data are spectral densities instead, integration must be revised.

See spectrum.csv, solves.json and report.json for all plotted values and assumptions.
