# Sideways incidence at 30 and 60 degrees

Run `.venv/bin/python -m acoustic_wall.check_angles_30_60`.

Uses the fixed selected expanded_300_75pct geometry: 300 mm period, 225 mm central clear opening, 360.2 mm depth. Angles are measured from the wall normal in the horizontal cross-section; both signs are checked. This does not address elevation-angle incidence.

Bloch-phase transverse boundaries and outgoing modal ports, with the incident transverse mode selected by its overlap with the oblique plane wave. Power is normalised by incident normal flux, including every propagating outgoing diffraction order. Sweep 700–1400 Hz every 25 Hz on the 1.5 mm interface-aligned mesh; 0.75 mm checks at 800, 1000 and 1250 Hz for ±30 and ±60 degrees. Additional empty-space and mirrored-geometry/sign-reversal checks at 60 degrees passed. Results are saved in report.json, runs.json and transmission.csv.

The weighted summary uses the same interpolated spectral envelope for each angle with trapezoidal sampling weights. It is a comparative screening metric, not full measured-band integration or an outdoor A-weighted noise rating. The model omits screens, losses, finite height/length, ground and supports. No reoptimisation occurred. The 75% frontal opening is not a claim of 75% view or airflow projected at oblique angles.
