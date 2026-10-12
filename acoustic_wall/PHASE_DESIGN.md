# Phase-guided design workflow

Adds isolated-channel complex transmission measurements, a nonzero ideal
amplitude target, explicit phase/amplitude geometry objectives, and diffraction
analysis. This does **not** reproduce the authors' exact COMSOL port model,
author CAD, thermoviscous simulations or experimental arrangement.

## Reproduce

Run in the repository root with fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest acoustic_wall.test_solver acoustic_wall.test_phase_design -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.phase_optimise --out acoustic_wall/results/my_phase_search
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.refine_phase acoustic_wall/results/my_phase_search --out acoustic_wall/results/my_phase_refinement
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.plot_phase_results acoustic_wall/results/my_phase_refinement/report.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.render_field --report acoustic_wall/results/my_phase_refinement/report.json --out acoustic_wall/results/my_phase_refinement/fields
```

Stages save machine-readable results. Search has a fixed random seed; local
refinement is deterministic. Existing hologram/ultrasound scripts are unchanged.

## Paper prescription

Yang et al. (2025), equations 2–3 and Figures 2–4:
https://www.nature.com/articles/s41598-025-04885-6

For openness O, relative to the central region:

```
phi_A - phi_centre = -(1+O)*pi/2
phi_C - phi_centre = +(1+O)*pi/2
```

At O=0.75 these are -157.5 and +157.5 degrees, modulo 360. A common overall
phase does not affect the target. The ribbed-channel geometry family is retained.

## Ideal amplitude target

A **prescribed-pressure, piecewise-constant aperture surrogate** estimates a
target amplitude ratio. At subwavelength pitch and normal incidence, the only
propagating diffraction order is the spatial mean. Its coefficient is
`sum(width_fraction_i * A_i * exp(i*phi_i))`.

Both side amplitudes are fixed at 1; central amplitude is optimised in [0,1].
This excludes the meaningless all-zero solution. A nominal 75/12.5/12.5 percent
split gives centre amplitude 0.308; actual 150/19/19 mm clear channels within
200 mm pitch give about 0.234.

This surrogate assigns zero pressure on the remaining wall footprint. That is
an ideal prescribed field, **not a physical rigid-wall boundary condition**.
It is only a target template; its apparent cancellation is never reported as
device transmission loss. It does not enforce passive realisability or capture
reflections/coupling. The full physical scattering solve supplies passive R/T.
Local resonant pressure can exceed one without implying acoustic gain.

## Single-channel characterisation

Extracted channels have rigid transverse duct boundaries and discrete cosine
modes for outgoing radiation. A unit fundamental mode enters. Complex S21 is
referenced to physical module faces by removing exterior uniform-duct phase.
The blank central duct supplies a phase reference for the side-channel curves.

The library sweeps 6 core depths and 15 slit widths at 1 mm resolution and
provides phase-matched seeds. Isolated responses are not substituted for the
coupled assembled cell response.

## Assembled-cell objective and measurements

At 1000 Hz, normal incidence:

```
0.55 * transmitted_power
+ 0.25 * mean((1-cos(relative_phase_error))/2 over side regions)
+ 0.20 * normalised_complex_pattern_error
```

The pattern error removes common complex scale. Vanishing fields receive a
penalty, not a reward. Explicit relative-phase loss prevents a small central
target amplitude from hiding an incorrect central phase.

Assembled regional measurements are **total exit-pressure averages** on the
first uniform-air plane, de-embedded half a grid spacing to the outlet. They
are neither isolated S21 nor independently driven ports. The field varies
within each region; matching three averages cannot guarantee the mechanism.

Search varies side slits 2–16 mm, core depth 130–260 mm and common guide length
20–120 mm, snapped to 1 mm. Shortlisted candidates are reranked and locally
refined on 0.5 mm meshes. Local slit adjustments can extend below 2 mm, provided
they remain grid-resolved. At least 75% central clear width is preserved; side
slits add sightline area. Top/bottom framing is not part of this 2-D model.

Diagnostics report all propagating orders' transmitted **power**, evanescent
pressure and decay lengths, and the first evanescent orders. The near-outlet
evanescent **pressure-squared fraction is not a power fraction** or absorption.

## Validation and limits

Report: 1, 0.5 and 1/3 mm grids; 950–1050 Hz in 25 Hz steps at 0/±30 degrees;
±0.5 mm dimension perturbations. The phase target is for normal incidence;
off-angle phase measurements are diagnostic only. This run does not optimise
the full measured yard spectrum or off-axis performance.

Screening thresholds are engineering checks, not paper claims. A field with
evanescent orders alone does not establish a **bound surface eigenmode**.
There is no dispersion-branch/eigenmode calculation here. Finite 3-D geometry,
thermoviscous losses, structural behaviour and experimental validation remain.

## Latest candidate and reproducible figures

The selected cell has a 200 mm repeat width, 150 mm central opening, 240 mm
ribbed core and 68 mm guides at each end (376 mm total depth). Side slits are
2 and 3 mm. A subsequent search over independent axial rib-train offsets did
not improve the objective; both offsets remain zero.

An independent 0.25 mm calculation predicts T = 0.00306 and 25.14 dB
transmission loss. The preceding 1/3 mm result differs by about 1.3 dB, so
the 1 dB convergence check is not satisfied. Relative regional exit-pressure
phases are approximately -174.75, 0 and -176.38 degrees: the prescribed
-157.5, 0 and +157.5 degree phase target is not satisfied within 15 degrees.
The strong, nearly balanced positive/negative first evanescent orders do not
establish a unidirectional bound surface wave.

`inspect_candidate.py` saves the quarter-mm field and independently fits
forward/backward fundamental channel waves in the exit guides. These modal
amplitudes are distinct from total outlet pressure averages. `finalize_phase.py`
combines that inspection with the frequency, angle and tolerance checks;
`plot_phase_results.py` plots the resulting report. `render_field.py` accepts
`--inspection-dir` to render the saved numerical field without re-solving.
Final artifacts are in `results/phase_guided_final/`.
