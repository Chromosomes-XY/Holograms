# 300 mm cell with two cavity sets

Run `.venv/bin/python -m acoustic_wall.two_stage_300`.

The central gap is 240 mm (80% true frontal opening); side half-bodies occupy 30 mm each. Periodic repetition joins neighbouring half-bodies into 60 mm slats. The final wall must retain >=75% clear area after frames/posts. Four separate straight cavities form two left/right pairs along the depth. Each cavity has independently configurable length, width, mouth height and start in the geometry builder. This initial bounded optimisation varies the four lengths and the separation between sets. Widths are fixed at 22 mm and mouths at 20 mm. No narrow rib structures are introduced.

The second set starts after the end of both first-set cavities plus the optimised separation. Total depth follows the last cavity end plus a 10 mm back cap; depth is not constrained to the earlier 180 mm. Minimum separation between same-side cavities is explicitly checked. The solver uses exact rectangular interfaces and enforces 80% opening on every evaluated mesh.

Fourteen-member differential evolution, five generations, 3 mm screening mesh; three finalists rechecked at 1.5 mm. Selected design swept at 25 Hz intervals on a 1.5 mm mesh and checked at 0.75 mm for 800, 1000 and 1250 Hz. Coarse search bounds and all dimensions are in report.json. The objective uses nine frequency samples with measured A-weighted band-power weights and a worst-frequency penalty; it is not a full band-integrated outdoor metric. history.jsonl is a task-specific cache; invalidate if model changes.

Normal-incidence rigid lossless 2D periodic acoustics, exact discrete modal ports. Transmission sums all propagating orders, including diffracted orders above the first 300 mm-period cut-on (approximately 1143 Hz). Screens, posts, elastic vibration, finite wall and thermoviscous losses are omitted. This is a preliminary design search, not a validated product prediction.
