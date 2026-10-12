# Discrete 5 mm manufacturing study

All manufactured geometric lengths are integer multiples of 5 mm. The computational unit is a complete 75 mm slat plus 225 mm clear gap, eliminating the old 37.5 mm half-slat convention from the manufacturing geometry. Pitch is 300 mm and straight-through frontal clearance is exactly 75%. Opposing recesses leave at least a 5 mm central web; same-side recesses have at least 10 mm separation. Front and rear solid margins are at least 10 mm. Search depth is 350–600 mm, not an assertion that deeper designs cannot help.

## Results

| Construction | Depth (mm) | Weighted power remaining | Reduction (dB) |
|---|---:|---:|---:|
| 4 covered recesses | 380 | 52.4% | 2.80 |
| 6 covered recesses | 430 | 47.6% | 3.22 |
| 4 open notches | 400 | 64.6% | 1.89 |
| 6 open notches | 415 | 68.4% | 1.65 |

These are normal-incidence, lossless 2D periodic-unit estimates. They are not directly comparable with the preceding point-source finite-wall receiver-region dB results. All supplied nonzero bands enter the objective. Assume uniform power per Hz within the bands, supplied ear gain held at each band value, and incoherent addition of power. The zero-power 20 kHz input needs no solve.

## Construction interpretation

For the covered recess, cut a fully accessible rectangular recess from each side, then add a short flat 5 mm cover strip over part of its length. The exposed end is a rectangular mouth. This reproduces the simulated pocket without requiring a one-piece hidden undercut. Covers must close the intended cavity boundary; fasteners, seals, vertical end closures and structural details have not been engineered. The orange areas in the drawing identify these separate strips.

The open-notch option removes those strips completely and opens each recess over its full length. It is easier to access and clean, but it changes the acoustic geometry substantially. The result applies to this tested family of side-opening rectangular notches, not every possible notch/slot design. All corners are ideal square corners; actual cutter radii must be added before fabrication validation.

Dimension CSVs describe the selected covered-recess and open-notch options, including true cut penetration, mouth length and cover-strip size. Every dimension is a multiple of 5 mm. Plotting-mesh steps are numerical choices and are not manufacturing tolerances.

## Search and checks

40 discrete mutation proposals per four/six covered-recess family, and 20 per open-notch family, plus initial seeds. This is a limited local comparison, not a global optimum. Five-point Gauss integration screens candidates; rejected candidates may stop only when their accumulated nonnegative objective already exceeds the incumbent. Finalists use a finer 2.5 mm maximum mesh and nine points per band. Dominant 630–1600 Hz bands are subsequently checked with 33 points; the winning covered-recess design additionally uses 65. The rest retain nine points. Search meshes use 5 mm below 4 kHz upper band edge, otherwise 2.5 mm. High-frequency bands have very small weight but finite resolution remains a limitation.

Separate checks compare 2.5 and 1.25 mm meshes at identical nine-point integration in the 800, 1000 and 1250 Hz bands. The best covered and uncovered options are also screened at 0, ±30 and ±60 degrees, using five-point band integration. Those angular numbers are less frequency-resolved than the final normal-incidence score; compare like sampling when comparing angles. No directional distribution was assumed, and oblique incidence was not included in the search objective. See pocket_checks.json and notch_checks.json.

Reproduce from the repository root with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.manufacturable_search
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.manufacturable_notches
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.check_manufacturable
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.finalize_manufacturable
```

The finalizer generates or reads winner_quadrature65.json. Caches are model-specific. The new geometries have not yet been run as a finite 9.9 m wall or with the intended 1.8 m height, realistic losses, source variation or weather exposure.
