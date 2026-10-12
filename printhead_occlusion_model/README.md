# Print-Head Occlusion Hologram Workflows

This folder models how an inserted print head modifies a holographic acoustic
pressure field.

## Layout

- `lib/occlusion_lib.py` contains the shared print-head occlusion model.
- `workflows/free_field_baseline/` generates a normal unobstructed hologram and
  renders how that saved hologram behaves under head occlusion.
- `workflows/occlusion_optimized/` optimizes directly across a sweep of
  print-head occlusion levels, renders the optimized result, compares against a
  free-field optimized hologram, and exports printable hologram height STLs.
- `workflows/three_inward_holograms/` is the same four-step optimized workflow,
  but with three circular holograms arranged around the central volume and
  pointing inward in a tilted three-face corner configuration.
- `workflows/legacy_demo/` keeps the earlier all-in-one sweep demo for reference.
- Each workflow owns a `results/` subfolder containing its generated `.npz`,
  `.png`, and `.stl` outputs.

## Main Optimized Workflow

Run these from the repository root:

```bash
.venv/bin/python printhead_occlusion_model/workflows/occlusion_optimized/01_optimize_for_occlusion_levels.py
.venv/bin/python printhead_occlusion_model/workflows/occlusion_optimized/02_render_occlusion_optimized_from_saved.py
.venv/bin/python printhead_occlusion_model/workflows/occlusion_optimized/03_compare_free_vs_occlusion_optimized.py
.venv/bin/python printhead_occlusion_model/workflows/occlusion_optimized/04_render_hologram_heights_and_export_stl.py
```

The optimizer writes:

```text
printhead_occlusion_model/workflows/occlusion_optimized/results/occlusion_optimized_vertical_line_hologram.npz
```

The render/export scripts read that same file and write their outputs back into
`printhead_occlusion_model/workflows/occlusion_optimized/results/`.

## Three Inward-Pointing Holograms

Run these from the repository root:

```bash
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/00_generate_non_optimized_spiral.py
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/02_render_non_optimized_from_saved.py
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/01_optimize_for_occlusion_levels.py
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/02_render_occlusion_optimized_from_saved.py
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/03_compare_free_vs_occlusion_optimized.py
.venv/bin/python printhead_occlusion_model/workflows/three_inward_holograms/04_render_hologram_heights_and_export_stl.py
```

This workflow writes to:

```text
printhead_occlusion_model/workflows/three_inward_holograms/results/
```

The default actuator geometry points at `(25, 0, 0) mm` from three mutually
orthogonal directions. The centres are offset downward from the target centre,
giving the rotated corner layout visible in the paper figure rather than a flat
ring of transducers.

## Baseline Workflow

```bash
.venv/bin/python printhead_occlusion_model/workflows/free_field_baseline/01_generate_unobstructed_vertical_line.py
.venv/bin/python printhead_occlusion_model/workflows/free_field_baseline/02_render_occluded_from_saved.py
.venv/bin/python printhead_occlusion_model/workflows/free_field_baseline/03_export_hologram_stls.py
.venv/bin/python printhead_occlusion_model/workflows/free_field_baseline/04_generate_cubic_reservoir_stl.py
```

Baseline outputs live under:

```text
printhead_occlusion_model/workflows/free_field_baseline/results/
```

The baseline uses two circular actuators, each 50 mm in diameter. The numerical
aperture remains a 50 mm square sampling grid, with samples outside the circular
active face masked out during initialization, optimization, propagation, and
occlusion visibility sampling.

The configured actuator positions represent the outer faces of the acrylic
walls. Each hologram is placed 5 mm outward, opposite its inward-facing normal,
and the 2.68 mm acrylic extends inward from the configured wall position. The
angular-spectrum transfer conserves transverse wavenumber across
the parallel interfaces and includes phase/refraction, angle-dependent acoustic
impedance transmission, and coherent internal reflections in the acrylic. The
outer side uses water (`1500 m/s`, `1000 kg/m³`), while the inner side currently
uses a hydrogel estimate (`1600 m/s`, assumed `1000 kg/m³`, or `1.60 MRayl`).
The inner-medium speed and density are separate step-01 parameters so measured
GelMA values can replace these estimates later.
The default PMMA values are `2750 m/s` and `1175 kg/m³` (about `3.2 MRayl`),
matching [measured PMMA acoustic properties](https://pmc.ncbi.nlm.nih.gov/articles/PMC10658616/).

Step 02 also writes `results/hologram_height_arrays.png`, showing the printable
height map for each saved actuator on a shared millimetre scale.

The third step converts the two saved phase fields into printable height maps
and writes one circular STL per actuator under `results/hologram_stls/`. Each
mesh has a small top identification tab: one raised dot marks PZT A and two
raised dots mark PZT B.

Step 04 generates an open-top `70 x 70 x 70 mm` cubic reservoir with `2.68 mm`
walls and base under `results/reservoir_stls/`. Its internal dimensions are
`64.64 x 64.64 x 67.32 mm`. Both dimensions can be overridden with the
`--outer-side-mm` and `--wall-thickness-mm` command-line arguments.

The legacy demos similarly write to
`printhead_occlusion_model/workflows/legacy_demo/results/`.

## Model Notes

The print head is represented as:

- an absorbing cylindrical shell, treated as occluding rather than transmitting;
- an open interior;
- a bottom fluid-air interface, initially flat and strongly reflective.

The current numerical model captures geometric shadowing / partial occlusion. It
does not yet solve full acoustic scattering, diffraction around the head, or
phase-shifted reflected waves.
