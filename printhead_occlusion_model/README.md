# Print-Head Partial Occlusion Model

This folder contains a first-pass model for how an inserted print head changes a
holographic acoustic field.

The model splits the print head into two acoustic components:

- An absorbing cylindrical shell. This is treated as occluding: acoustic energy
  intercepted by the shell is lost rather than transmitted through the head.
- A bottom fluid-air interface. This is treated as strongly reflective and has a
  radially symmetric shape profile. The initial profile is flat everywhere.

For each PZT and target voxel, the model computes whether the straight acoustic
path from the PZT centre to the voxel intersects either print-head component.
The path is then attenuated by a soft transmission factor:

- `1.0` means clear path.
- `shell_transmission` means a path blocked by the absorbing shell.
- `bottom_transmission` means a path blocked by the reflective fluid-air bottom.
- `shell_soft_edge_m` and `bottom_soft_edge_m` control soft transition regions
  around the component boundaries.

This captures geometric shadowing / partial occlusion. It does not yet model
phase-shifted reflections, diffraction around the head, or full acoustic
scattering. The bottom already carries a configurable complex pressure
reflection coefficient for the next modelling step.

Run:

```bash
.venv/bin/python printhead_occlusion_model/demo_partial_occlusion.py
```

Output:

```text
printhead_occlusion_model/partial_occlusion_demo.png
```
