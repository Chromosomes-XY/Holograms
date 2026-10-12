# Independent left/right smooth resonator optimisation

Run `.venv/bin/python -m acoustic_wall.optimise_smooth_pair`.

One 200 mm periodic cell has a left side body x=0–20 mm, a right side body x=180–200 mm, and a 160 mm unobstructed central gap. Each side has one smooth rectangular cavity opening sideways into this same gap. Periodic repetition joins half-bodies into 40 mm slats. The two cavities are independently dimensioned and coupled through the full Helmholtz solution, not solved as isolated resonators. Translating the previous centre-slat geometry into this convention preserves the baseline (15 mm passages and a combined 4 mm separator across the periodic seam).

Each side independently varies cavity length (50–110 mm), mouth start (10–65 mm from front), mouth height (10–30 mm), and cavity width (8–15 mm). Wall depth remains 180 mm, inner mouth wall 3 mm. All rectangle edges align with the numerical grid. At least 2 mm outer wall remains on each periodic half; together these make a >=4 mm separator. No internal ribs. 80% clear opening enforced per cell, leaving an allowance for whole-wall supports; final whole-wall opening must remain >=75%.

Fixed seed, 16-member differential evolution and six generations on a 2.5 mm maximum mesh. Three finalists checked at 1 mm; best fine objective selected. Subsequent 25 Hz sweep at 1 mm and selected 0.5 mm checks. Objective weights come from the supplied A-weighted measured band powers, distributed over three frequencies around each of 800/1000/1250 Hz, with a worst-frequency penalty. This is a finite sampled screening objective, not an exact broadband integral. See report.json for every parameter and result. history.jsonl is the evaluation cache; invalidate it if the model changes.

Rigid lossless normal-incidence 2D periodic pressure acoustics; outgoing modal axial ports. Screens, losses, posts and finite-height effects absent. No claims of global optimality or validated yard noise reduction.
