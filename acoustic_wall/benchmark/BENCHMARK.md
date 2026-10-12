# Rectangular PGUOM benchmark

**Outcome: the paper's physical Figure 5 results are not yet reproduced.**
The numerical solver passes conservation and independent assembly checks, but
that is not evidence that this reconstructed geometry is the authors' model.
No geometry was fitted to the Figure 5 curves.

## Comparison

![Reference comparison](/Users/Edward/Documents/GitHub/Holograms/acoustic_wall/benchmark/benchmark_comparison.png)

| Nominal openness | Local T at 2000 Hz | Local loss (dB) | Spectrum T RMSE | Compared frequencies |
|---|---:|---:|---:|---:|
| 0.40 | 0.0044061 | 23.56 | 0.120 | 23 |
| 0.47 | 1.6896e-06 | 57.72 | 0.156 | 27 |
| 0.53 | 0.085524 | 10.68 | 0.236 | 24 |
| 0.60 | 0.0012425 | 29.06 | 0.333 | 26 |
| 0.67 | 0.10202 | 9.91 | 0.470 | 30 |
| 0.73 | 0.18733 | 7.27 | 0.549 | 28 |

The paper reports T < 0.1 at 2000 Hz for all six cases. Reference spectra are
colour-digitised from the supplied Figure 5a screenshot, not author raw data.
Approximate reading uncertainty is 0.02 in T and 3 Hz away from steep features.
Overlapped/obscured pixels are omitted. RMSE compares observed reference points
within 3 Hz of local sampled frequencies; it is not a fitted-curve score.
See `reference_digitisation_check.png` and `reference_provenance.json`.

## What is benchmarked

1. Exact agreement of the new nonuniform finite-volume assembly with the original
   solver on the same uniform grid. Empty and closed guides; reciprocity; scaling;
   conservation; independently driven inlet ports; mirror symmetry from S1.
2. The paper's Table 1 phase prescription and amplitudes in three separately
   excited incoming waveguide ports. Zero-thickness rigid separators end at a
   common guide; 40 mm isolated entrance and 110 mm common downstream region,
   total transverse width 75 mm. Incoming power is summed across all three
   ports. This is an explicitly specified reconstruction, not the authors'
   unprovided COMSOL port setup. It confirms the expected amplitude-tuning trend.
3. Table 2 single-channel complex transmission, referenced to an empty channel
   between the same planes. COMSOL's +iωt phasors are conjugated for our -iωt
   convention. Table 2 channel numbering mirrors the Figure 3 example; its order
   is preserved. No per-case sign or phase adjustment is made.
   Figure 5 labels its pressure scale only as max/min; pressure-colour agreement
   is therefore qualitative. Our fields use one fixed incident-normalised scale.
4. Six fixed physical geometries with published slit widths and waveguide lengths,
   1700–2300 Hz in 20 Hz steps, and 2000 Hz pressure fields. Hard side walls are
   the primary reconstruction. Periodic boundaries, alternate wall-width
   accounting and rib end placement are separate sensitivity checks.

![Diagnostics](/Users/Edward/Documents/GitHub/Holograms/acoustic_wall/benchmark/benchmark_diagnostics.png)

## Geometry and mesh assumptions

Published H=85.75 mm, n=10 rib pairs, axial rib width h=5 mm, longitudinal
wall thickness tt=2 mm. Each side uses the Table 2 slit width. Per-side guide
lengths are 0.3H, 0.2H, 0.1H, 0.5H, 0.5H and 0.5H (Figure 5 caption).
The main reconstruction places equally spaced rib centres at (j+1/2)H/n.
The axial end setback is inferred, not separately dimensioned by the source.

Main width convention: Wi spans wall centrelines; clear air width is Wi−2 mm.
Thus nominal openness is not the actual clear fraction: the 0.73 case has
53/75 = 70.67% central clearance in this reconstruction. The alternative treats
Wi as clear air and adds walls, giving an 81 mm repeat. Both conventions are
recorded; neither is asserted to be verified CAD. Do not use nominal openness
as a finished-wall transparency measurement.

Mesh lines coincide with all geometry edges. The 0.58 and 0.53 mm slits remain
distinct at every mesh level. Maximum-cell meshes of 0.4, 0.2 and 0.1 mm are
checked, plus grids with 50, 25 and 12.5 micrometre cells near rectangle edges.
The 0.73 case additionally uses 6.25 micrometre edge cells.
Mesh steps here are local acoustic discretisation, not manufacturing tolerances.
All propagating modal power is counted, including higher duct modes above cutoff.
Outgoing modal boundaries replace PMLs for these homogeneous rigid/periodic guides.

Maximum |R+T−1| across saved physical solves: 1.24e-11.
Maximum relative linear-system residual: 5.89e-11.
Neither metric measures geometry fidelity or continuum mesh convergence.

## Consequence for the garden-wall design

The previous approximately 25 dB prediction belongs to an unvalidated 2D
candidate. This benchmark does not validate that number or establish the paper's
mechanism in that candidate. First resolve the Table 2 complex-transmission
mismatch and geometry/port definitions; then require agreement across the six
spectra and mesh levels before optimising a 75%-clear 1000 Hz wall.

Source files and provenance: `sources/paper.pdf`, `sources/supplement.pdf`.
Paper: https://doi.org/10.1038/s41598-025-04885-6
COMSOL phasor convention:
https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.002.html

See `README.md` for commands. All raw solve metrics and saved 2000 Hz fields
are in `results/`; the benchmark summary is `report.json`.
