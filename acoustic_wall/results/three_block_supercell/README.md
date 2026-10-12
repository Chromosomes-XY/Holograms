# Three-block cavity supercell

A 900 mm-wide repeating unit consists of three 300 mm blocks. Each retains a 75 mm solid slat and 225 mm clear passage, so the complete ABC supercell has 675 mm (75%) projected opening before supports. A 9.9 m wall would contain 11 ABC repeats, or 33 individual blocks.

The source design is the six-covered-cavity slat, not the angled solid-beam experiment. Every block has six cavities, giving 18 cavities per supercell. All cavity dimensions are multiples of 5 mm; minimum solid walls remain 5 mm. All three slats share a common outer depth, determined by the deepest cavity layout.

This first bounded coordinate search uses six parameters: independent axial cavity-length/gap scale for each block, and independent mouth-width offsets for each block. Cavity widths and starting offsets retain the original design. Rounded dimensions are checked against the original manufacturing constraints. This does not independently optimise all 72 geometric dimensions.

One coupled Helmholtz solve models the entire 900 mm supercell with periodic lateral boundaries and outgoing modal ports. All propagating transmitted modes are included. It does not average isolated A/B/C transmission curves. Repeating three identical cells was checked against the original 300 mm model at 800, 1000 and 1250 Hz, agreeing within 1e-7 in transmitted power.

Objective: measured band power times supplied A-weighting gain times band-averaged power transmission. Unknown within-band spectra are assumed uniform in power per Hz. The weak bands above 3150 Hz are conservatively assigned unity transmission for every design; report.json records their total weight. Search uses 5 mm mesh and 3/5 frequency quadrature; finalists use 2.5 mm mesh and 5/17 quadrature, with the selected comparison using 33 nodes in 630–1600 Hz bands. This is a limited search, not a global optimum or exhaustive mesh-convergence study.

The previous approximately 3.1 dB cavity-versus-filled comparison used a finite wall and a central receiver region. Results here are periodic plane-wave transmitted power and cannot be subtracted directly from that 3.1 dB result. Rigid, lossless, 2D normal incidence excludes over-wall sound, wall ends, ground, supports and weather losses. A later 33-block finite-wall comparison is needed to determine yard performance.

Run: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.three_block_supercell`

## Initial result

21 unique screened candidates. Selected scales A=90%, B=90%, C=110%; mouth offsets 0, −5, 0 mm. Rounded dimensions are in cavity_dimensions_mm.csv. Shared depth 470 mm. Final weighted transmission: original 430 mm A–A–A 48.1248%; 470 mm A–A–A 47.6226%; 470 mm A–B–C 46.0947%. Mixed tuning improves 1.53 percentage points (approximately 0.14 dB) versus the matching-depth control. Relative to the original-depth control, improvement is 2.03 points (0.19 dB). Improvement is modest, mostly near 1 kHz; substantial bandwidth broadening has not been established. 17-to-33-node dominant-band refinement changes selected transmission by 0.057 percentage points. No additional finer spatial mesh study was performed for the mixed design. The unchanged high-frequency tail is 0.7496% of input weighted power.
