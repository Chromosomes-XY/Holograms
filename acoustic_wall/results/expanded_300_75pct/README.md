# Expanded 300 mm, 75%-open optimisation

Run `.venv/bin/python -m acoustic_wall.expand_300`.

300 mm periodic pitch; opaque half-bodies 37.5 mm each; 225 mm central gap (75%). Four smooth cavities independently vary length, width and mouth height, with independent first-mouth positions and same-side inter-cavity gaps: 16 variables. Minimum cavity width 14 mm, mouth height 10 mm, inner wall 3 mm, outer half-wall at least 3 mm, same-side partition at least 10 mm. Total depth follows the longest side plus a 10 mm cap. Supports must lie within the opaque budget: no further blockage allowance remains.

24-member differential evolution, nine generations, deterministic seed and 3 mm mesh, with cached evaluations. Objective: 85% measured-spectrum-weighted mean transmission plus 15% worst transmission, at 15 samples around the 800, 1000 and 1250 Hz bands. Four finalists and the unchanged 75% seed are re-ranked on a 10 Hz grid plus training nodes at 1.5 mm. This dense selection uses an explicitly interpolated weighting envelope, not an exact band integration. Checks at 0.75 mm use the original 15 nodes. A fair previous-80%-open comparison uses the same dense grid and 1.5 mm mesh. Cache is model-specific; invalidate on model changes.

Rigid lossless 2D periodic pressure acoustics with exact discrete modal radiation ports. All propagating diffraction orders included. No screens, losses, structural vibration, finite height or ground. Numerical mesh checks do not establish outdoor validity. See report.json for all geometry and comparison data; history.jsonl for the search and cache/ for individual solves.

Additional spot checks: `python -m acoustic_wall.check_expanded_angles` evaluates −30°, −15°, 0°, +15°, +30° at 800, 1000 and 1250 Hz on the same 1.5 mm interface-aligned mesh. This uses Bloch seam phases and a weighted Hermitian modal-port decomposition, selects incidence by plane-wave overlap, and sums all outgoing propagating modes. The search itself was normal-incidence only. See bloch_verification.json, angle_checks.json and geometry_mm.csv.
