"""Generate a 4 x 4 cm acoustic hologram for a thin S-shaped focal line.

The target is a 20 x 20 mm S drawn in the plane 30 mm in front of the plate.
It is represented as a smooth narrow band rather than a one-pixel stroke so it
remains resolvable at the 2.05 MHz operating frequency.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hologramGen import (
    acoustic_params,
    design_hologram_single_plane,
    export_hologram_stl,
    impedance_params,
    make_grid,
    make_square_pzt,
    project_hologram,
)


# Plate and focal-plane geometry
# 256² keeps the exported ASCII STL manageable while sampling the 0.8 mm
# stroke at more than four grid cells across.
N = 256
SIMULATION_WIDTH_M = 0.045
PZT_SIDE_M = 0.040
FOCAL_DISTANCE_M = 0.030
# Keep a 10 mm margin around the S within the 40 x 40 mm region.
S_BOUND_M = 0.020
LINE_WIDTH_M = 0.80e-3
N_OPTIMIZATION_ITERATIONS = 300

# Hologram and acoustic material parameters
MIN_THICKNESS_M = 0.4e-3
MAX_THICKNESS_M = 3.0e-3
FREQUENCY_HZ = 2.05e6
C_MEDIUM = 1500.0
C_HOLOGRAM = 2550.0

OUTPUT_STEM = "s_line_4cm_f30mm_hologram"


def make_s_line_target(xv, yv, bound_m, line_width_m):
    """Return a smooth, centred S stroke inside a square focal-plane bound."""
    if bound_m <= 0.0 or line_width_m <= 0.0:
        raise ValueError("S bound and line width must be positive.")

    # A cubic Bezier path gives a deliberately simple, legible single-stroke S.
    # Two connected curves trace the whole letter, from its upper-right
    # terminal through the centre crossover to its lower-left terminal.
    anchors = np.array(
        [
            [0.68, 1.00],   # upper-right terminal
            [-1.00, 1.08],  # upper-left sweep
            [-1.00, 0.30],
            [0.00, 0.00],   # centre crossover
            [1.00, -0.30],
            [1.00, -1.08],  # lower-right sweep
            [-0.68, -1.00], # lower-left terminal
        ]
    )
    t = np.linspace(0.0, 1.0, 500)
    segments = []
    for start, control1, control2, end in (
        anchors[:4],
        anchors[3:],
    ):
        curve = (
            (1.0 - t)[:, None] ** 3 * start
            + 3.0 * (1.0 - t)[:, None] ** 2 * t[:, None] * control1
            + 3.0 * (1.0 - t)[:, None] * t[:, None] ** 2 * control2
            + t[:, None] ** 3 * end
        )
        segments.append(curve)

    # Normalise the Bezier control coordinates so the whole stroke is bounded
    # by ``bound_m`` rather than allowing control values above one to spill out.
    points = np.vstack(segments)
    points *= bound_m / (2.0 * np.max(np.abs(anchors)))
    distance2 = np.full_like(xv, np.inf, dtype=float)
    for point_x, point_y in points:
        distance2 = np.minimum(distance2, (xv - point_x) ** 2 + (yv - point_y) ** 2)

    sigma = line_width_m / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    target = np.exp(-0.5 * distance2 / sigma**2)
    return target * ((np.abs(xv) <= bound_m / 2.0) & (np.abs(yv) <= bound_m / 2.0))


def save_preview(xv, yv, thickness, target, projected, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fields = (
        (thickness * 1e3, "Hologram thickness", "viridis", "thickness [mm]"),
        (target, "Target: thin S at 30 mm", "inferno", "target amplitude"),
        (np.abs(projected), "Simulated field at 30 mm", "inferno", "amplitude"),
    )
    for axis, (field, title, cmap, label) in zip(axes, fields):
        image = axis.pcolormesh(xv * 1e3, yv * 1e3, field, shading="auto", cmap=cmap)
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_xlim(-20, 20)
        axis.set_ylim(-20, 20)
        fig.colorbar(image, ax=axis, label=label)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    output_dir = Path(__file__).resolve().parent
    stl_output = output_dir / f"{OUTPUT_STEM}.stl"
    preview_output = output_dir / f"{OUTPUT_STEM}.png"

    _, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    p0_amp = make_square_pzt(xv, yv, PZT_SIDE_M)
    target = make_s_line_target(xv, yv, S_BOUND_M, LINE_WIDTH_M)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

    thickness, history = design_hologram_single_plane(
        p0_amp=p0_amp,
        target_at_z=target,
        z_target=FOCAL_DISTANCE_M,
        dx=dx,
        lam_m=acoustic["lam_m"],
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
        n_iter=N_OPTIMIZATION_ITERATIONS,
        Tmin=MIN_THICKNESS_M,
        Tmax=MAX_THICKNESS_M,
    )
    projected = project_hologram(
        p0_amp, thickness, dx, acoustic["lam_m"], FOCAL_DISTANCE_M,
        impedance["Zt"], impedance["Zh"], impedance["Zm"], acoustic["kh"], acoustic["km"],
    )
    facets, nonmanifold_edges = export_hologram_stl(thickness, dx, stl_output)
    save_preview(xv, yv, thickness, target, projected, preview_output)

    print("Focal distance [mm]:", FOCAL_DISTANCE_M * 1e3)
    print("S target bound [mm]:", S_BOUND_M * 1e3, "x", S_BOUND_M * 1e3)
    print("S line FWHM [mm]:", LINE_WIDTH_M * 1e3)
    print("Final amplitude correlation:", history["amp_corr"][-1])
    print(f"Saved STL to {stl_output} ({facets} facets; {nonmanifold_edges} nonmanifold edges)")
    print(f"Saved preview to {preview_output}")


if __name__ == "__main__":
    main()
