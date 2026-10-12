# Filled-rectangle comparison

Run `.venv/bin/python -m acoustic_wall.compare_filled_array` after the finite-wall and dominant-band refinement runs.

The comparison fills every cavity with rigid material while preserving the external slat envelope, 9.9 m array width, 373.4 mm depth, 300 mm pitch and 225 mm clear gap (75%). Interior shared slats are 75 mm wide; the finite array retains the same 37.5 mm half-slats at the two outer ends. This is an open array of solid rectangles, not a completely closed solid wall.

Both geometries use identical grid edges, source position, PML, rigid boundary conditions and frequency sampling. Existing no-wall and cavity fields are reused. Bands through 1600 Hz are explicitly solved; the five dominant bands use the refined mesh. The remaining 5.1% of supplied weighted input is added identically as unattenuated background. Band powers add incoherently. No absolute SPL is inferred.

In the difference maps, positive dB means the cavity array is quieter than the filled rectangles; negative means it is louder locally. Receiver-region comparisons integrate mean-square pressure over x=-4..4 m, z=0.7..3.5 m before taking the dB ratio. They are not transmitted flux or an outdoor wall rating. All prior finite-array limitations apply, including 2D line-source behaviour, no height/ground, band-centre sampling and incomplete mesh convergence.
