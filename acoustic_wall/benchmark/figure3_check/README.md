# Figure 3 fixed-dimension check

This run uses the explicitly published Figure 3 dimensions, without optimisation:

| Quantity | Value |
|---|---:|
| Frequency | 2000 Hz |
| Sound speed | 343 m/s |
| Clear channel width | 25 mm |
| Core depth H | 85.75 mm |
| Rib count | 10 opposing pairs |
| Axial rib width h | 5 mm |
| Side slits | 2.00 and 3.25 mm |
| Blank channel slit | 25 mm |
| Wall thickness | 2 mm, represented by the rigid air-domain boundary for an isolated channel |
| Extrusion t | 30 mm in the paper; invariance assumed in this 2D check |
| Added guides | None in the isolated-channel test; published assembled guides are 34.3 mm per end |

The blank opening is explicitly w2=W2=25 mm in the caption. This check therefore
uses a 25 mm clear air domain, rather than the 23 mm air domain of the earlier
wall-inclusive benchmark. The complete three-channel assembly is not rerun here.

**Exactness limit:** the source does not independently dimension the axial gaps
and end setbacks. This calculation uses ten equally spaced rib centres at
(j+1/2)H/10. That produces 3.575 mm internal air gaps and 1.7875 mm end setbacks.
Those placements are assumptions. This is not certified exact author CAD.
The earlier full-spectrum reconstruction remains available as an assumption
audit, not a reproduction of the paper.

## Independent propagation check

The existing conservative finite-volume solver was compared with a separately
implemented linear-triangle finite-element solver using consistent element mass
matrices, natural rigid boundaries and modal radiation ports. The FEM transverse
port basis is obtained from a separate 1D finite-element eigenproblem. It does
not reuse the FV stencil, mass matrix or eigenvectors.

Both methods use the same physical rectangles and successively smaller cells
near corners (50, 25 and 12.5 micrometres; maximum interior cell size 0.3 mm).
Phases are relative to an empty channel between each method's own port planes;
COMSOL's +iωt convention is conjugated to our -iωt convention before comparison.
Table 2's channel ordering is used; the Figure 3 example labels the sides in the
opposite order.

| Slit | Paper amplitude | FV amplitude | FEM amplitude | Paper relative phase | FV relative phase | FEM relative phase |
|---|---:|---:|---:|---:|---:|---:|
| 3.25 mm | 0.69 | 0.88578 | 0.88252 | +120.6° | +146.08° | +145.59° |
| 25 mm blank | 1 | 1.00000 | 1.00000 | 0° | 0.00002° | −0.00109° |
| 2.00 mm | 0.49 | 0.45633 | 0.45911 | −120.6° | −107.03° | −107.32° |

For the ribbed channels, method differences are below 0.0033 in amplitude and
0.49° in phase. These are much smaller than the discrepancies with the paper,
especially for the 3.25 mm slit. Agreement supports the numerical calculation
for the stated geometry; it does not prove the shared geometric or physical
assumptions match the paper.

Three independent FEM tests pass: empty-guide phase/power, closed-guide
reflection, and reciprocity/scaling with an obstacle. Reciprocity is tested by
reversing excitation on the same mesh, rather than rebuilding a reflected mesh
with differently oriented triangles.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest acoustic_wall.test_independent_fem
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.check_figure3
```

The numerical comparisons, exact mesh coordinates and fields are saved alongside
this file. No candidate optimisation or fitted parameter values were used.
Source: https://www.nature.com/articles/s41598-025-04885-6, Figure 3 and Table 2.
