"""Evaluate a one-grid-cell 10 mm ring target at several focal depths."""

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
    project_hologram,
)
from sweep_10mm_ring_width import radial_average, ring_fwhm


FOCAL_DISTANCES_M = np.array((10, 20, 30, 40, 50), dtype=float) * 1e-3
RING_RADIUS_M = 5e-3
MAX_HOLOGRAM_THICKNESS_M = 4e-3
OUTPUT_DIR = Path("zero_width_ring_depth_sweep")


def make_one_cell_ring(xv, yv, radius_m, cell_width_m):
    radius = np.sqrt(xv**2 + yv**2)
    return (np.abs(radius - radius_m) <= cell_width_m / 2.0).astype(float)


def simulate_depth(focal_distance_m, target, x, xv, yv, dx, pzt, acoustic, impedance):
    thickness, history = design_hologram_single_plane(
        p0_amp=pzt,
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
        Tmax=MAX_HOLOGRAM_THICKNESS_M,
    )
    amplitude = np.abs(
        project_hologram(
            pzt,
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
    )
    radial_m, radial_amp = radial_average(amplitude, xv, yv, dx / 2.0)
    width_m, peak_radius_m, _, _ = ring_fwhm(radial_m, radial_amp, RING_RADIUS_M)
    annulus = np.abs(np.sqrt(xv**2 + yv**2) - peak_radius_m) <= dx
    angular_cv = float(np.std(amplitude[annulus]) / np.mean(amplitude[annulus]))
    return {
        "focal_distance_m": focal_distance_m,
        "thickness": thickness,
        "amplitude": amplitude,
        "radial_m": radial_m,
        "radial_amp": radial_amp,
        "width_m": width_m,
        "peak_radius_m": peak_radius_m,
        "angular_cv": angular_cv,
        "peak_amp": float(amplitude.max()),
        "correlation": float(history["amp_corr"][-1]),
    }


def save_field_grid(results, x, effective_width_m):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes = axes.ravel()
    for axis, result in zip(axes, results):
        normalized = result["amplitude"] / result["amplitude"].max()
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
            linewidths=0.9,
        )
        axis.set_xlim(-7, 7)
        axis.set_ylim(-7, 7)
        axis.set_aspect("equal")
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("y [mm]")
        axis.set_title(
            f"F={result['focal_distance_m'] * 1e3:.0f} mm; "
            f"FWHM={result['width_m'] * 1e3:.3f} mm; "
            f"CV={result['angular_cv'] * 100:.1f}%"
        )
    axes[-1].axis("off")
    fig.colorbar(image, ax=axes[:-1].tolist(), label="normalised pressure amplitude")
    fig.suptitle(
        "10 mm diameter one-cell ring target "
        f"(effective target width {effective_width_m * 1e3:.3f} mm)",
        fontsize=16,
    )
    fig.savefig(OUTPUT_DIR / "zero_width_ring_focal_plane_fields.png", dpi=200)
    plt.close(fig)


def save_radial_profiles(results):
    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for result in results:
        offset_mm = (result["radial_m"] - RING_RADIUS_M) * 1e3
        normalized = result["radial_amp"] / result["radial_amp"].max()
        axis.plot(
            offset_mm,
            normalized,
            linewidth=1.8,
            label=(
                f"F={result['focal_distance_m'] * 1e3:.0f} mm; "
                f"FWHM={result['width_m'] * 1e3:.3f} mm"
            ),
        )
    axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(-2, 2)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("radial offset from 5 mm ring radius [mm]")
    axis.set_ylabel("normalised pressure amplitude |p|")
    axis.set_title("One-cell ring target: radial profiles by focal depth")
    axis.legend()
    fig.savefig(OUTPUT_DIR / "zero_width_ring_radial_profiles.png", dpi=220)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    pzt = make_circular_pzt(xv, yv, PZT_SIDE_M)
    target = make_one_cell_ring(xv, yv, RING_RADIUS_M, dx)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

    results = []
    for focal_distance_m in FOCAL_DISTANCES_M:
        print(f"Designing focal depth {focal_distance_m * 1e3:.0f} mm...", flush=True)
        result = simulate_depth(
            focal_distance_m, target, x, xv, yv, dx, pzt, acoustic, impedance
        )
        results.append(result)
        focal_mm = int(round(focal_distance_m * 1e3))
        np.savez_compressed(
            OUTPUT_DIR / f"zero_width_ring_f{focal_mm:02d}mm.npz",
            x_m=x,
            effective_target_width_m=dx,
            target_ring=target,
            thickness_m=result["thickness"],
            focal_pressure_amplitude=result["amplitude"],
            radial_coordinate_m=result["radial_m"],
            radial_pressure_amplitude=result["radial_amp"],
            simulated_fwhm_m=result["width_m"],
            peak_radius_m=result["peak_radius_m"],
            angular_coefficient_of_variation=result["angular_cv"],
        )
        print(
            f"  FWHM={result['width_m'] * 1e3:.4f} mm; "
            f"angular CV={result['angular_cv'] * 100:.2f}%",
            flush=True,
        )

    save_field_grid(results, x, dx)
    save_radial_profiles(results)
    with (OUTPUT_DIR / "zero_width_ring_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "focal_depth_mm",
                "effective_target_width_mm",
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
                    result["focal_distance_m"] * 1e3,
                    dx * 1e3,
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
