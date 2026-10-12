"""Find the minimum simulated width of a 10 mm diameter acoustic ring."""

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
)
from hologramGen import (
    acoustic_params,
    design_hologram_single_plane,
    impedance_params,
    make_circular_pzt,
    make_grid,
    make_ring_target,
    project_hologram,
)


FOCAL_DISTANCE_M = 20e-3
RING_DIAMETER_M = 10e-3
RING_RADIUS_M = RING_DIAMETER_M / 2.0
TARGET_WIDTHS_M = np.array((0.05, 0.10, 0.20, 0.30, 0.40, 0.50)) * 1e-3
MAX_HOLOGRAM_THICKNESS_M = 4.0e-3
OUTPUT_DIR = Path("ring_10mm_diameter_width_sweep")


def interpolate_crossing(coords, values, start, direction, level):
    index = start
    while 0 <= index + direction < values.size:
        next_index = index + direction
        if values[next_index] <= level:
            x1, x2 = coords[index], coords[next_index]
            y1, y2 = values[index], values[next_index]
            if y1 == y2:
                return float(x2)
            return float(x1 + (level - y1) * (x2 - x1) / (y2 - y1))
        index = next_index
    return float("nan")


def ring_fwhm(radius_m, radial_profile, expected_radius_m):
    search = np.abs(radius_m - expected_radius_m) <= 1.5e-3
    indices = np.flatnonzero(search)
    peak_index = indices[int(np.argmax(radial_profile[search]))]
    half_level = 0.5 * radial_profile[peak_index]
    inner = interpolate_crossing(radius_m, radial_profile, peak_index, -1, half_level)
    outer = interpolate_crossing(radius_m, radial_profile, peak_index, 1, half_level)
    return outer - inner, radius_m[peak_index], inner, outer


def radial_average(field, xv, yv, bin_width_m):
    radius = np.sqrt(xv**2 + yv**2)
    bin_index = np.floor(radius / bin_width_m).astype(int)
    count = np.bincount(bin_index.ravel())
    total = np.bincount(bin_index.ravel(), weights=field.ravel())
    valid = count > 0
    centres = (np.arange(count.size) + 0.5) * bin_width_m
    profile = np.zeros_like(total)
    profile[valid] = total[valid] / count[valid]
    return centres[valid], profile[valid]


def simulate_width(target_width_m, x, xv, yv, dx, pzt_amp, acoustic, impedance):
    target = make_ring_target(xv, yv, RING_RADIUS_M, target_width_m)
    thickness, history = design_hologram_single_plane(
        p0_amp=pzt_amp,
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
        Tmax=MAX_HOLOGRAM_THICKNESS_M,
    )
    amplitude = np.abs(
        project_hologram(
            pzt_amp,
            thickness,
            dx,
            acoustic["lam_m"],
            FOCAL_DISTANCE_M,
            impedance["Zt"],
            impedance["Zh"],
            impedance["Zm"],
            acoustic["kh"],
            acoustic["km"],
        )
    )
    radial_m, radial_amp = radial_average(amplitude, xv, yv, dx / 2.0)
    width_m, peak_radius_m, inner_m, outer_m = ring_fwhm(
        radial_m, radial_amp, RING_RADIUS_M
    )
    annulus = np.abs(np.sqrt(xv**2 + yv**2) - peak_radius_m) <= dx
    angular_cv = float(np.std(amplitude[annulus]) / np.mean(amplitude[annulus]))
    return {
        "target_width_m": target_width_m,
        "thickness": thickness,
        "amplitude": amplitude,
        "radial_m": radial_m,
        "radial_amp": radial_amp,
        "width_m": width_m,
        "peak_radius_m": peak_radius_m,
        "inner_m": inner_m,
        "outer_m": outer_m,
        "angular_cv": angular_cv,
        "peak_amp": float(amplitude.max()),
        "correlation": float(history["amp_corr"][-1]),
    }


def save_overlay(results):
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for result in results:
        radial_offset_mm = (result["radial_m"] - RING_RADIUS_M) * 1e3
        normalized = result["radial_amp"] / result["radial_amp"].max()
        axis.plot(
            radial_offset_mm,
            normalized,
            linewidth=1.8,
            label=(
                f"target {result['target_width_m'] * 1e3:.2f} mm; "
                f"simulated {result['width_m'] * 1e3:.3f} mm"
            ),
        )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(-2.0, 2.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("radial offset from 5 mm ring radius [mm]")
    axis.set_ylabel("normalised pressure amplitude |p|")
    axis.set_title("10 mm diameter ring at 20 mm focal distance")
    axis.legend()
    fig.savefig(OUTPUT_DIR / "ring_width_radial_profiles.png", dpi=220)
    plt.close(fig)


def save_best_field(best, x):
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    normalized = best["amplitude"] / best["amplitude"].max()
    image = axis.imshow(
        normalized,
        extent=(x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3),
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axis.contour(
        x * 1e3,
        x * 1e3,
        normalized,
        levels=(0.5,),
        colors=("cyan",),
        linewidths=1.0,
    )
    axis.set_xlim(-7, 7)
    axis.set_ylim(-7, 7)
    axis.set_aspect("equal")
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_title(
        f"Narrowest ring: amplitude FWHM {best['width_m'] * 1e3:.3f} mm"
    )
    fig.colorbar(image, ax=axis, label="normalised pressure amplitude")
    fig.savefig(OUTPUT_DIR / "narrowest_ring_field.png", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    pzt_amp = make_circular_pzt(xv, yv, PZT_SIDE_M)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

    results = []
    for target_width_m in TARGET_WIDTHS_M:
        print(f"Designing target width {target_width_m * 1e3:.2f} mm...", flush=True)
        result = simulate_width(
            target_width_m, x, xv, yv, dx, pzt_amp, acoustic, impedance
        )
        results.append(result)
        width_um = int(round(target_width_m * 1e6))
        np.savez_compressed(
            OUTPUT_DIR / f"ring_target_width_{width_um:04d}um.npz",
            x_m=x,
            target_width_m=target_width_m,
            thickness_m=result["thickness"],
            focal_pressure_amplitude=result["amplitude"],
            radial_coordinate_m=result["radial_m"],
            radial_pressure_amplitude=result["radial_amp"],
            simulated_fwhm_m=result["width_m"],
            peak_radius_m=result["peak_radius_m"],
            angular_coefficient_of_variation=result["angular_cv"],
        )
        print(
            f"  simulated FWHM={result['width_m'] * 1e3:.4f} mm; "
            f"angular CV={result['angular_cv']:.4f}",
            flush=True,
        )

    best = min(results, key=lambda result: result["width_m"])
    save_overlay(results)
    save_best_field(best, x)
    with (OUTPUT_DIR / "ring_width_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_width_mm",
                "simulated_amplitude_fwhm_mm",
                "peak_ring_radius_mm",
                "angular_coefficient_of_variation",
                "peak_pressure_amplitude_arb",
                "target_correlation",
                "maximum_hologram_thickness_mm",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result["target_width_m"] * 1e3,
                    result["width_m"] * 1e3,
                    result["peak_radius_m"] * 1e3,
                    result["angular_cv"],
                    result["peak_amp"],
                    result["correlation"],
                    np.nanmax(result["thickness"]) * 1e3,
                )
            )
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
