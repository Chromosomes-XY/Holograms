"""Compare a 40 mm circular-PZT point focus at several distances.

This reuses the single-plane point target and corrected hologram model.
"""

from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_square_pzt_hologram import (
    C_HOLOGRAM,
    C_MEDIUM,
    FREQUENCY_HZ,
    MIN_THICKNESS_M,
    N,
    N_OPTIMIZATION_ITERATIONS,
    PZT_SIDE_M,
    SIMULATION_WIDTH_M,
    TARGET_DOT_DIAMETER_M,
)
from hologramGen import (
    acoustic_params,
    apply_hologram,
    design_hologram_single_plane,
    impedance_params,
    make_center_dot_target,
    make_circular_pzt,
    make_grid,
    project_hologram,
    propagate_ASM,
)


FOCAL_DISTANCES_M = (10e-3, 20e-3, 30e-3, 40e-3, 50e-3)
N_AXIAL_SAMPLES = 181
TRANSVERSE_PLOT_LIMIT_M = 3e-3
OUTPUT_DIR = Path("tweezer_focal_distance_comparison")
DESIGN_MAX_THICKNESS_M = 4.0e-3
OUTPUT_PREFIX = "circular_pzt_4mm"


def crossing_coordinate(coords, values, peak_index, direction, level):
    """Linearly interpolate one threshold crossing away from a peak."""
    index = peak_index
    while 0 <= index + direction < values.size:
        next_index = index + direction
        if values[next_index] <= level:
            x1, x2 = coords[index], coords[next_index]
            y1, y2 = values[index], values[next_index]
            if y2 == y1:
                return float(x2)
            return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
        index = next_index
    return float("nan")


def profile_width(coords, values, relative_level):
    """Return full width around the central main-lobe peak."""
    centre = values.size // 2
    search_radius = max(2, int(round(2e-3 / abs(coords[1] - coords[0]))))
    lower = max(0, centre - search_radius)
    upper = min(values.size, centre + search_radius + 1)
    peak_index = lower + int(np.argmax(values[lower:upper]))
    level = float(values[peak_index]) * relative_level
    left = crossing_coordinate(coords, values, peak_index, -1, level)
    right = crossing_coordinate(coords, values, peak_index, 1, level)
    return float(right - left), peak_index, left, right


def simulate_case(focal_distance_m, x, xv, yv, dx, pzt_amp, acoustic, impedance):
    target = make_center_dot_target(xv, yv, TARGET_DOT_DIAMETER_M)

    thickness, history = design_hologram_single_plane(
        p0_amp=pzt_amp,
        target_at_z=target,
        z_target=focal_distance_m,
        dx=dx,
        lam_m=acoustic["lam_m"],
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
        n_iter=N_OPTIMIZATION_ITERATIONS,
        Tmin=MIN_THICKNESS_M,
        Tmax=DESIGN_MAX_THICKNESS_M,
    )
    focal_field = project_hologram(
        pzt_amp,
        thickness,
        dx,
        acoustic["lam_m"],
        focal_distance_m,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )
    focal_amp = np.abs(focal_field)
    centre = N // 2
    x_profile = focal_amp[centre, :]
    y_profile = focal_amp[:, centre]
    x_fwhm_amp, _, _, _ = profile_width(x, x_profile, 0.5)
    y_fwhm_amp, _, _, _ = profile_width(x, y_profile, 0.5)

    p_exit = apply_hologram(
        np.asarray(pzt_amp, dtype=np.complex128),
        thickness,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )
    z_samples = np.linspace(0.5 * focal_distance_m, 1.25 * focal_distance_m, N_AXIAL_SAMPLES)
    xz_amp = np.empty((z_samples.size, N), dtype=float)
    on_axis_amp = np.empty(z_samples.size, dtype=float)
    for z_index, z_value in enumerate(z_samples):
        plane_amp = np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z_value)))
        xz_amp[z_index] = plane_amp[centre, :]
        on_axis_amp[z_index] = plane_amp[centre, centre]

    focal_window = (z_samples >= 0.8 * focal_distance_m) & (
        z_samples <= 1.15 * focal_distance_m
    )
    focal_indices = np.flatnonzero(focal_window)
    axial_peak_index = focal_indices[int(np.argmax(on_axis_amp[focal_window]))]
    axial_peak_z_m = float(z_samples[axial_peak_index])
    peak_x_profile = xz_amp[axial_peak_index]
    peak_x_fwhm_amp, _, _, _ = profile_width(x, peak_x_profile, 0.5)
    target_z_index = int(np.argmin(np.abs(z_samples - focal_distance_m)))
    axial_peak_to_target_ratio = float(
        on_axis_amp[axial_peak_index] / on_axis_amp[target_z_index]
    )

    return {
        "focal_distance_m": focal_distance_m,
        "thickness": thickness,
        "focal_amp": focal_amp,
        "x_profile": x_profile,
        "y_profile": y_profile,
        "z_samples": z_samples,
        "xz_amp": xz_amp,
        "on_axis_amp": on_axis_amp,
        "x_fwhm_amp_m": x_fwhm_amp,
        "y_fwhm_amp_m": y_fwhm_amp,
        "axial_peak_z_m": axial_peak_z_m,
        "peak_x_fwhm_amp_m": peak_x_fwhm_amp,
        "axial_peak_to_target_ratio": axial_peak_to_target_ratio,
        "peak_focal_amp": float(focal_amp.max()),
        "final_corr": float(history["amp_corr"][-1]),
    }


def save_case(result, x):
    focal_mm = int(round(result["focal_distance_m"] * 1e3))
    np.savez_compressed(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_f{focal_mm:02d}mm.npz",
        x_m=x,
        z_m=result["z_samples"],
        thickness_m=result["thickness"],
        focal_pressure_amplitude=result["focal_amp"],
        xz_pressure_amplitude=result["xz_amp"],
        on_axis_pressure_amplitude=result["on_axis_amp"],
        x_fwhm_amplitude_m=result["x_fwhm_amp_m"],
        y_fwhm_amplitude_m=result["y_fwhm_amp_m"],
        axial_peak_z_m=result["axial_peak_z_m"],
        peak_x_fwhm_amplitude_m=result["peak_x_fwhm_amp_m"],
        axial_peak_to_target_amplitude_ratio=result["axial_peak_to_target_ratio"],
    )


def save_comparison_plot(results, x):
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    limit = TRANSVERSE_PLOT_LIMIT_M * 1e3
    for result in results:
        focal_mm = result["focal_distance_m"] * 1e3
        axis.plot(
            x * 1e3,
            result["x_profile"] / result["x_profile"].max(),
            linewidth=1.7,
            label=(
                f"F={focal_mm:.0f} mm; "
                f"FWHM={result['x_fwhm_amp_m'] * 1e3:.3f} mm"
            ),
        )
    axis.axhline(0.5, color="gray", linewidth=1, linestyle="--")
    axis.set_xlim(-limit, limit)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("transverse position [mm]")
    axis.set_ylabel("normalised pressure amplitude |p|")
    axis.set_title("40 mm circular PZT; hologram thickness range 0.4–4.0 mm")
    axis.legend()
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_overlay_profiles.png", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    pzt_amp = make_circular_pzt(xv, yv, PZT_SIDE_M)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

    results = []
    for focal_distance_m in FOCAL_DISTANCES_M:
        print(f"Designing focal distance {focal_distance_m * 1e3:.0f} mm...", flush=True)
        result = simulate_case(
            focal_distance_m, x, xv, yv, dx, pzt_amp, acoustic, impedance
        )
        save_case(result, x)
        results.append(result)
        print(
            "  amplitude FWHM [mm]: "
            f"x={result['x_fwhm_amp_m'] * 1e3:.4f}, "
            f"y={result['y_fwhm_amp_m'] * 1e3:.4f}; "
            f"axial peak={result['axial_peak_z_m'] * 1e3:.3f} mm",
            flush=True,
        )

    save_comparison_plot(results, x)
    with (OUTPUT_DIR / f"{OUTPUT_PREFIX}_focal_distance_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_focal_distance_mm",
                "x_amplitude_fwhm_mm",
                "y_amplitude_fwhm_mm",
                "axial_peak_position_mm",
                "peak_position_x_amplitude_fwhm_mm",
                "axial_peak_to_target_amplitude_ratio",
                "peak_focal_pressure_amplitude_arb",
                "final_mean_target_correlation",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result["focal_distance_m"] * 1e3,
                    result["x_fwhm_amp_m"] * 1e3,
                    result["y_fwhm_amp_m"] * 1e3,
                    result["axial_peak_z_m"] * 1e3,
                    result["peak_x_fwhm_amp_m"] * 1e3,
                    result["axial_peak_to_target_ratio"],
                    result["peak_focal_amp"],
                    result["final_corr"],
                )
            )
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
