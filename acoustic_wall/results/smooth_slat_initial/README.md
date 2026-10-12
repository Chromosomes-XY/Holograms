# Smooth-cavity slat initial simulation

Run: `.venv/bin/python -m acoustic_wall.smooth_slat_initial`.

Fixed geometry is the previous drawn concept, translated into a periodic cell: slat x=80–120 mm; depth z=0–180 mm; first cavity x=83–98, z=20–127 mm, mouth through x=80–83, z=20–40 mm; second cavity x=102–117, z=60–146 mm, mouth x=117–120, z=60–80 mm. All rectangle interfaces are exact mesh edges. No dimensions were optimised. Geometric cavity lengths are not end-corrected acoustic lengths.

80% straight-through opening per repeat; final posts/rails must preserve at least 75% overall. Normal plane incidence, rigid lossless 2D Helmholtz, periodic transverse boundaries and discrete outgoing modal axial ports. No insect screens, thermal/viscous losses, structural vibration, finite-height or ground effects. Power transmission includes every propagating outgoing mode. The control is the identical slat with both cavities filled in, on the same mesh.

Coarse sweeps at 50 Hz spacing over 600–1600 Hz, supplemented by 10 Hz samples from 720–1000 Hz, selected 1 mm checks, and a 0.5 mm check at 1000 Hz. Modal continuation visualises exterior pressure without changing the port solution. See runs.json and transmission.csv for all results, report.json for dimensions/assumptions, field_*.npz for complex fields. Cache must be invalidated if geometry/solver changes. These runs are a screening study, not a validated outdoor attenuation prediction.
