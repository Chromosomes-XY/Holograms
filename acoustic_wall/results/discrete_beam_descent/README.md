# Two-layer solid-beam discrete descent

Run: `OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.discrete_beam_descent`

Each 300 mm periodic cell contains two independently angled solid rectangles. Beam-centre positions vary on a 100 mm lattice; front centre (150,200) mm is a coordinate reference, rear centres are x=50/150/250 mm and z=300/400/500/600 mm. Front translation does not alter normal-incidence infinite-periodic power transmission. Rear versus front relative positions are the physical degrees of freedom.

Stock lengths and thicknesses are multiples of 5 mm. Search length steps initially 25 mm, followed by a bounded 5 mm refinement. Thickness steps 5 mm, angle steps 5 degrees. Minimum thickness 20 mm, axial gap 10 mm. No cavities.

First-layer projected opening >=75%; combined straight-through opening >=50% (an exploratory assumption following relaxation of the original 75% requirement). Combined opening is the complement of the exact union of periodic projected beam intervals, not the product of the layer openings. Posts and rails are excluded.

Three-start discrete coordinate descent tests positive and negative neighbours and accepts the lowest objective. Three sweeps per start bound computation. This is not continuous gradient descent and does not establish a global or fully converged local optimum.

Objective: sum of measured band power times supplied A-weighting gain times band-averaged power transmission, divided by the unattenuated weighted sum. All propagating outgoing modes count. Power assumed uniform per Hz within each supplied band, since no within-band recordings are available.

Search: 5 mm Cartesian mesh, three Gauss nodes per band through 3150 Hz; higher-band transmission conservatively set to one. Finalists: 2.5 mm mesh and denser frequency integration, then 33 nodes in dominant 630–1600 Hz bands and 9 elsewhere for selected best. A separate 1.25 mm versus 2.5 mm comparison at matching quadrature estimates mesh sensitivity.

Rigid, lossless, 2D Helmholtz finite-volume solver with periodic lateral boundaries and discrete outgoing modal ports. Beam surfaces use stair-step geometry. This excludes finite wall ends, height, supports, ground, absorption, oblique sound and structural feasibility. Energy balance alone does not demonstrate spatial convergence. Geometric opening and rasterised opening differ slightly.

## Result

88 screening evaluations. Selected 230 × 40 mm at +5 degrees and 225 × 40 mm at −30 degrees, aligned laterally, centres 300 mm apart in depth. Physical beam envelope 523.7 mm; computational depth parameter 620 mm includes empty air margins. First-layer clear 80.0%, combined clear 51.0%. Final weighted power transmission 0.681025 (1.668 dB reduction) versus starting layout 0.876048. At matching frequency quadrature, 2.5 mm grid gives 0.673225 and 1.25 mm gives 0.674212: 0.099 percentage point difference. Denser frequency quadrature remains material (~1 percentage point from 17 to 33 dominant-band nodes). These are approximate periodic-model results, not a global optimum or finite-wall performance guarantee.
