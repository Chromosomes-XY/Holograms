# Reproducible paper benchmark

The newer [Figure 3 fixed-dimension check](figure3_check/README.md) uses a 25 mm
clear channel and compares the original finite-volume method with an independent
triangle finite-element solver. The methods agree closely for that geometry,
while disagreement with the paper remains. Rib end positions are still inferred;
neither reconstruction should be described as exact author CAD.

Run commands from the repository root using the existing `.venv`. Numerical
routines require NumPy/SciPy; plots require Matplotlib/Pillow. No COMSOL licence
or proprietary model is included.

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest acoustic_wall.test_solver acoustic_wall.test_phase_design acoustic_wall.test_benchmark
.venv/bin/python -m acoustic_wall.digitize_reference
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.benchmark_ports
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .2
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .1 --frequencies 2000
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --corner-mm .05 --frequencies 2000
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --corner-mm .025
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .2 --corner-mm .0125 --frequencies 2000
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .2 --corner-mm .00625 --frequencies 2000 --labels .73
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --frequencies 2000 --boundary periodic
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --frequencies 2000 --convention added
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --frequencies 2000 --layout end_anchored
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.run_paper_benchmark --step-mm .4 --corner-mm .025 --frequencies 2000 --boundary periodic
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.audit_paper_channels
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.report_paper_benchmark
```

Every solve is cached in `results/`, so interrupted runs can resume. Cache names
encode all geometry interpretations, boundary type, mesh and frequency. Delete
or move matching result files manually before rerunning after a numerical-method
change; existing files are not automatically invalidated.

The original solver, optimised wall geometry and old results are preserved.
`benchmark_mesh.py` adds an independently assembled nonuniform finite-volume
solver and multiple incoming ports. `paper_benchmark.py` contains transcribed
Table 2 values and explicit geometry assumptions. `benchmark_ports.py` uses
Table 1 phase/amplitude values. Source PDFs, the user-supplied figure and source
hashes are in `sources/`.

This benchmark covers the rectangular part of the supplied figure. The
cylindrical device requires an appropriate axisymmetric/3D model and is not
represented by a rectangular 2D cross section.
