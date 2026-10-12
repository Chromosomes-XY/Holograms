"""Evaluate saved point-focus holograms 0.5 mm before their focal planes."""

from pathlib import Path
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_tweezer_focal_distances import profile_width
from generate_square_pzt_hologram import (
    C_HOLOGRAM,
    C_MEDIUM,
    FREQUENCY_HZ,
    N,
    PZT_SIDE_M,
    SIMULATION_WIDTH_M,
)
from hologramGen import (
    acoustic_params,
    apply_hologram,
    impedance_params,
    make_grid,
    make_square_pzt,
    propagate_ASM,
)


INPUT_DIR = Path("tweezer_focal_distance_comparison")
OFFSET_M = 0.5e-3
OUTPUT_STEM = INPUT_DIR / "tweezer_prefocal_0p5mm"
PLOT_LIMIT_M = 3e-3


def evaluate_saved_case(path, x, dx, pzt_amp, acoustic, impedance):
    saved = np.load(path)
    focal_distance_m = float(path.stem.removeprefix("tweezer_f").removesuffix("mm")) * 1e-3
    evaluation_distance_m = focal_distance_m - OFFSET_M
    thickness = saved["thickness_m"]
    focal_amp = saved["focal_pressure_amplitude"]

    exit_field = apply_hologram(
        np.asarray(pzt_amp, dtype=np.complex128),
        thickness,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )
    prefocal_amp = np.abs(
        propagate_ASM(exit_field, dx, acoustic["lam_m"], evaluation_distance_m)
    )
    centre = N // 2
    prefocal_x = prefocal_amp[centre]
    prefocal_y = prefocal_amp[:, centre]
    focal_x = focal_amp[centre]

    x_amp_fwhm_m = profile_width(x, prefocal_x, 0.5)[0]
    y_amp_fwhm_m = profile_width(x, prefocal_y, 0.5)[0]
    x_int_fwhm_m = profile_width(x, prefocal_x, np.sqrt(0.5))[0]
    y_int_fwhm_m = profile_width(x, prefocal_y, np.sqrt(0.5))[0]
    centre_pressure_ratio = float(prefocal_amp[centre, centre] / focal_amp[centre, centre])
    peak_pressure_ratio = float(prefocal_amp.max() / focal_amp.max())

    output_path = INPUT_DIR / f"tweezer_f{int(round(focal_distance_m * 1e3)):02d}mm_prefocal_0p5mm.npz"
    np.savez_compressed(
        output_path,
        x_m=x,
        evaluation_distance_m=evaluation_distance_m,
        pressure_amplitude=prefocal_amp,
        x_amplitude_fwhm_m=x_amp_fwhm_m,
        y_amplitude_fwhm_m=y_amp_fwhm_m,
        x_intensity_fwhm_m=x_int_fwhm_m,
        y_intensity_fwhm_m=y_int_fwhm_m,
        centre_pressure_ratio_to_focus=centre_pressure_ratio,
        peak_pressure_ratio_to_focus=peak_pressure_ratio,
    )
    return {
        "focal_distance_m": focal_distance_m,
        "evaluation_distance_m": evaluation_distance_m,
        "prefocal_amp": prefocal_amp,
        "prefocal_x": prefocal_x,
        "focal_x": focal_x,
        "x_amp_fwhm_m": x_amp_fwhm_m,
        "y_amp_fwhm_m": y_amp_fwhm_m,
        "x_int_fwhm_m": x_int_fwhm_m,
        "y_int_fwhm_m": y_int_fwhm_m,
        "centre_pressure_ratio": centre_pressure_ratio,
        "peak_pressure_ratio": peak_pressure_ratio,
    }


def save_plot(results, x):
    fig, axes = plt.subplots(len(results), 2, figsize=(10, 12), constrained_layout=True)
    limit_mm = PLOT_LIMIT_M * 1e3
    for row, result in enumerate(results):
        prefocal_norm = result["prefocal_amp"] / result["prefocal_amp"].max()
        axes[row, 0].imshow(
            prefocal_norm,
            extent=(x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3),
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        axes[row, 0].set_xlim(-limit_mm, limit_mm)
        axes[row, 0].set_ylim(-limit_mm, limit_mm)
        axes[row, 0].set_xlabel("x [mm]")
        axes[row, 0].set_ylabel("y [mm]")
        axes[row, 0].set_title(
            f"{result['evaluation_distance_m'] * 1e3:.1f} mm plane "
            f"(target F={result['focal_distance_m'] * 1e3:.0f} mm)"
        )

        focal_peak = result["focal_x"].max()
        axes[row, 1].plot(
            x * 1e3,
            result["prefocal_x"] / focal_peak,
            label="F − 0.5 mm",
        )
        axes[row, 1].plot(
            x * 1e3,
            result["focal_x"] / focal_peak,
            "--",
            label="F",
        )
        prefocal_half_level = 0.5 * result["prefocal_x"].max() / focal_peak
        axes[row, 1].axhline(
            prefocal_half_level,
            color="steelblue",
            linestyle=":",
            linewidth=1,
            label="prefocal half maximum",
        )
        axes[row, 1].set_xlim(-limit_mm, limit_mm)
        axes[row, 1].set_ylim(0, 1.05)
        axes[row, 1].set_xlabel("x [mm]")
        axes[row, 1].set_ylabel("|p| / focal-plane peak")
        axes[row, 1].set_title(
            f"Prefocal FWHM {result['x_amp_fwhm_m'] * 1e3:.3f} mm; "
            f"centre pressure {result['centre_pressure_ratio'] * 100:.1f}%"
        )
        axes[row, 1].legend()
    fig.savefig(f"{OUTPUT_STEM}_comparison.png", dpi=200)
    plt.close(fig)


def main():
    x, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    pzt_amp = make_square_pzt(xv, yv, PZT_SIDE_M)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)
    paths = sorted(INPUT_DIR.glob("tweezer_f[0-9][0-9]mm.npz"))
    if not paths:
        raise FileNotFoundError(f"No saved holograms found in {INPUT_DIR}")
    results = [
        evaluate_saved_case(path, x, dx, pzt_amp, acoustic, impedance)
        for path in paths
    ]
    save_plot(results, x)

    with Path(f"{OUTPUT_STEM}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_focal_distance_mm",
                "evaluated_plane_mm",
                "x_amplitude_fwhm_mm",
                "y_amplitude_fwhm_mm",
                "x_intensity_fwhm_mm",
                "y_intensity_fwhm_mm",
                "centre_pressure_ratio_to_focus",
                "peak_pressure_ratio_to_focus",
            )
        )
        for result in results:
            writer.writerow(
                (
                    result["focal_distance_m"] * 1e3,
                    result["evaluation_distance_m"] * 1e3,
                    result["x_amp_fwhm_m"] * 1e3,
                    result["y_amp_fwhm_m"] * 1e3,
                    result["x_int_fwhm_m"] * 1e3,
                    result["y_int_fwhm_m"] * 1e3,
                    result["centre_pressure_ratio"],
                    result["peak_pressure_ratio"],
                )
            )
    print(f"Saved prefocal evaluation to {OUTPUT_STEM}_comparison.png")


if __name__ == "__main__":
    main()
