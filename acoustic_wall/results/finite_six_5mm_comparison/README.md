# Best six-covered-cavity slat versus solid-filled rectangles: finite 2D array

This run uses the selected manufacturable six-cavity geometry from `manufacturable_5mm/report.json`. Every construction dimension is a multiple of 5 mm. Wall depth is 430 mm, slat width 75 mm, clear gap 225 mm and pitch 300 mm. There are 33 complete units across a nominal 9.9 m array. The finite sequence starts with a complete slat and ends with the corresponding clear gap; both compared geometries have identical ends. The source is at x=0, z=-3 m, measured from the wall front z=0.

The solid-filled comparison closes all six recesses with rigid material without changing the slat envelope, spacing or depth. It is an array of solid slats with 75% frontal opening, **not a completely closed wall**. A third no-wall solve supplies the reference. All three cases use identical coordinate grids, source forcing, absorbing boundaries and frequency samples.

## Method and spectral assumptions

Explicit total-field finite-volume Helmholtz simulation in a horizontal 2D section, with sound-hard solids and stretched-coordinate PML. The point source in 2D represents a vertical line source in 3D. The intended 1.8 m height, over-top diffraction, ground, material vibration and realistic losses are not modelled. Covers are assumed acoustically sealed wherever solid is shown.

Every supplied band through 1600 Hz is solved at its nominal centre, covering 94.8503% of the measured A-weighted input power. Higher bands are included as unchanged analytical background in each approximate whole-spectrum map. This is an explicit approximation, not a rigorous bound or a claim that the actual wall passes those bands unchanged. Frequency powers add incoherently as measured power × supplied ear gain × mean-square pressure. Band power is not multiplied by bandwidth again.

Bands from 630 through 1600 Hz use the finer mesh (10 mm maximum step near the wall, 20 mm away; 719,128 cavity-case air unknowns). Lower bands use 20/40 mm maxima (216,324 cavity-case air unknowns). Exact interfaces preserve 5 mm covers and all cavity dimensions. The present meshes and centre-frequency sampling are not a convergence demonstration. In particular, these finite-array estimates must not be equated with the more densely band-integrated periodic-cell optimisation scores.

The source strength is referenced to the analytical 2D field at 1 m. Absolute measured source SPL was not supplied; plotted levels are relative, not calibrated outdoor dBA. Receiver-region numbers integrate mean-square pressure over x=-4..4 m and z=0.7..3.5 m, then take the dB ratio. They are not transmitted acoustic power flux or a universal wall rating. Local enhancement is possible through interference.

## Outputs

- `comparison_maps.png`: filled and cavity weighted pressure maps, their broadband difference, and the corresponding 1 kHz fields/difference. Blue in a difference panel means cavities are quieter.
- `comparison_curves.png`: receiver-region reduction by sampled band, plus a weighted reduction profile approximately 2 m behind the rear face.
- `report.json`: geometry, assumptions and numerical comparison.
- `regional_band_comparison.csv`: receiver-region integrals and cavity advantage by sampled frequency.
- `broadband.npz`: weighted powers on a common plotting grid; dominant-band power sampled from the finer mesh.
- Per-case NPZ files: complex pressure fields and solver residuals.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.finite_six_comparison
```

The manifest checks geometry/source identity before reusing cached fields. Use a new output directory or explicitly invalidate caches if the solver or numerical settings change. Earlier four-cavity results are retained in `finite_wall_10m/`; their different depth and end construction mean they are not an otherwise-identical control for this new geometry.
