# Open acoustic wall: local research solver

The [paper benchmark](benchmark/BENCHMARK.md) now compares fixed published
rectangular designs with Figure 5, tests exact slit dimensions on nonuniform
meshes, and audits phase conventions and boundary conditions. **Paper
reproduction has not been established.** The earlier optimised wall's results
remain unvalidated screening predictions.

The newer [phase-guided workflow](PHASE_DESIGN.md) adds channel phase/amplitude
characterisation, explicit target matching, modal diagnostics and fine-grid
reoptimisation. The original direct search below remains a baseline.

This is an executable **2-D, rigid, lossless screening model**, not a validated
prediction for a garden wall. It solves the internal pressure field and optimises
physical geometry. It does not assign arbitrary independent phase/amplitude
values to a transparent screen and call the result a passive device.

## Run

From the Holograms repository root, using the existing environment:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest acoustic_wall.test_solver -v
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run evaluate --out acoustic_wall/results/baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run optimise --iterations 3 --popsize 4 --angles 0,30,-30 --out acoustic_wall/results/search_new
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run evaluate --paper-seed --out acoustic_wall/results/paper_seed_new
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.verify_candidate acoustic_wall/results/search_new/report.json --out acoustic_wall/results/search_new_verification
```

Optimisation output directories must be fresh. Results include a JSON report,
JSONL candidate history, best-so-far checkpoint, complex field NPZ and summary
PNG. `--geometry path/to/report.json` loads a previously selected geometry.
`--spacing-mm 1` re-evaluates at finer resolution. `--frequencies 800,900,1000,1100,1250`
and `--angles 0,15,30,-30` control operating conditions. Negative angle lists
starting with a minus sign should use `--angles=-30,0,30`.

## Geometry and paper provenance

Yang et al., *Phase gradient ultra open metamaterials for broadband acoustic
silencing*, Scientific Reports 15, 21434 (2025):
https://www.nature.com/articles/s41598-025-04885-6
Figure 3: https://www.nature.com/articles/s41598-025-04885-6/figures/3

The model uses the inspected Figure 3 topology: two channels with regularly
spaced opposing rigid ribs and a central clear passage. Continuous separators
extend into entrance and exit waveguides. Adjacent periods share a seam wall.
Each ribbed channel has a straight slit of a different width. Uniform ribs are
our reconstruction of the schematic; there is no supplied author CAD here.

**Coordinate correction:** the paper's H is along propagation, not the vertical
height of an installed fence. Its t is out-of-plane extrusion. A geometric
scale factor of two takes the 2 kHz reference to 1 kHz in lossless acoustics:
core H = 171.5 mm, guide extension 68.6 mm on each end, total 308.7 mm. The
out-of-plane t becomes 60 mm. A 75 mm deep square-frame illustration from the
conversation is not a scaled paper geometry.

`paper_scaled_seed()` preserves those nominal core/guide/rib/slit dimensions.
It explicitly adds separator walls to nominal channel widths: the full pitch
is 162 mm, not the paper's nominal 150 mm after scaling. This convention must
be resolved against author CAD/supplementary data before claiming reproduction.
The raster also snaps dimensions to its grid. No reproduction claim is made.

The default design instead uses 200 mm pitch and reserves 75% of that pitch
for the central clear passage; ribs, separators and guides occupy the rest.
The small side slits are also visible straight through, so total line-of-sight
openness can exceed 75%. Both quantities are reported. This is an **at-least
75% 2-D openness constraint**, not exactly 75% finished-wall area. Top/bottom
frames, posts, joints and the omitted extrusion dimension need a 3-D budget.

## Numerical method

Solve `laplacian(p) + (2*pi*f/c)^2 p = 0` in connected air pixels, with c=343 m/s,
using the exp(-i*omega*t) convention. Missing faces against solid cells have
zero flux (rigid Neumann walls). Transverse periodicity includes the Bloch
phase for oblique incidence. The first and last rows are homogeneous air.

For transverse wavenumber kx, the discrete axial phase q satisfies

```
4 sin(q/2)^2 / dz^2 = k^2 - 4 sin(kx*dx/2)^2 / dx^2.
```

The outward ghost row is the spectral operator with eigenvalues exp(i*q).
This discrete Dirichlet-to-Neumann equivalent admits outgoing propagating and
decaying evanescent orders without a finite absorbing layer. The entrance
includes a known incident-wave source. Reflected amplitudes are separated from
total pressure. Power uses sin(q)/dz flux weights summed over **all propagating
orders**, not mean pressure or one arbitrary quiet point. R+T=1 is checked for
every candidate. Near-field |p| can exceed one without violating passivity.

`Result.downstream()` is a one-transverse-dimension angular-spectrum propagator
for the **outgoing** field, using the same grid dispersion. It can export maps
beyond the final plane and preserves evanescent decay. The existing
`hologramGen.propagate_ASM` is a continuous-dispersion, two-transverse-dimension
homogeneous propagator; its principle is reused, but that water/ultrasound
workflow is not imported or modified. A 3-D exit plane could later use it with
air wavelength after ensuring outgoing-field separation and matching grids.

## Optimisation

Seeded differential evolution varies left slit, right slit, core length and
guide length. Parameters snap to the grid, duplicate candidates are cached,
and invalid/unresolved features are rejected. No gradient is claimed through
stair-step geometry. The objective is

`0.7 * mean(T over frequencies and angles) + 0.3 * max(T)`.

Default frequencies are 900, 1000 and 1100 Hz with equal weights; the measured
yard spectrum has **not** been digitised or used. The objective can be expanded
once calibrated spectral data and relevant source directions are supplied.
A short search is only a feasibility experiment, not a global optimum.
Always run `verify_candidate` before interpreting an optimised notch. It checks
2/1/0.5 mm meshes, sweeps 800–1250 Hz at 25 Hz intervals and perturbs four
dimensions by ±1 mm. Even that sweep may miss an extremely narrow resonance.

## Verification and required next steps

Automated checks cover empty-domain transmission and phase propagation,
closed-wall reflection, reciprocity, scale invariance, multi-order flux
conservation, geometric resolution and convergence for a rectangular obstacle.
These verify numerics, not agreement with the paper or experiment.

Before fabrication:

1. Resolve the original channel/wall dimensions and reproduce published
   transmission and phase curves under the same boundary conditions.
2. Refine selected candidates on 1 mm and 0.5 mm grids (or better), sweep a dense
   frequency/angle set and test dimensional perturbations. Coarse-grid peaks
   must not be reported as manufactured performance.
3. Add thermoviscous narrow-slit losses and finite wall compliance; a lossless
   resonance can be unrealistically strong or narrow. Wavelength resolution
   alone does not resolve viscous/thermal boundary layers.
4. Verify the finite extruded cell in 3-D, including connecting frames and all
   oblique incidence directions, then the finite panel and edge diffraction.
5. Compare a small printed panel against measurements before a full wall.

There is no airflow model, structural design, weathering model, calibrated
yard SPL prediction, manufacturing quote, or outdoor insertion-loss guarantee.

## First local experiment

The saved `results/initial_search` explores 64 candidates at 900/1000/1100 Hz
and 0/±30 degrees. `results/target_1000_search` explores 96 candidates at 1000 Hz
and the same angles. Neither is a global optimum. The latter candidate is
200 mm pitch, 216 mm core depth, 74 mm guides at each end (364 mm total),
10/4 mm side slits, 4 mm separators, and ten 10 mm ribs per side channel.
Its central opening is 75%; total line-of-sight opening is 82% in this 2-D model.

Normal-incidence 1000 Hz transmission loss changes from 9.44 dB on a 2 mm
grid to 1.87 dB on a 1 mm grid and 1.80 dB on a 0.5 mm grid. Thus the original
coarse-grid attenuation must **not** be treated as the candidate's performance.
The saved verification sweep and ±1 mm perturbations demonstrate the next
selection step: rank candidates on finer grids and across frequencies before
accepting a design. This is a working solver/optimiser, not a successful final
wall design or a replicated paper result.
