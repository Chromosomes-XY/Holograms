"""Create a detailed evaluation of the corrected 20 mm focal plane."""

from pathlib import Path
import argparse
import csv

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_tweezer_focal_distances import profile_width


INPUT_PATH = Path("tweezer_focal_distance_comparison/tweezer_f20mm.npz")
OUTPUT_DIR = INPUT_PATH.parent
IMAGE_PATH = OUTPUT_DIR / "tweezer_f20mm_focal_plane_evaluation.png"
SUMMARY_PATH = OUTPUT_DIR / "tweezer_f20mm_focal_plane_evaluation.csv"
PLOT_LIMIT_M = 2.5e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--image", type=Path, default=IMAGE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--title", default="Corrected 20 mm focal-plane field")
    args = parser.parse_args()

    saved = np.load(args.input)
    x_m = saved["x_m"]
    amplitude = saved["focal_pressure_amplitude"]
    centre = x_m.size // 2
    x_amplitude = amplitude[centre, :]
    y_amplitude = amplitude[:, centre]
    x_amp_fwhm_m, _, x_amp_left, x_amp_right = profile_width(x_m, x_amplitude, 0.5)
    y_amp_fwhm_m, _, _, _ = profile_width(x_m, y_amplitude, 0.5)

    amplitude_norm = amplitude / amplitude.max()
    x_amp_norm = x_amplitude / x_amplitude.max()
    y_amp_norm = y_amplitude / y_amplitude.max()
    limit_mm = PLOT_LIMIT_M * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    common_extent = (
        x_m[0] * 1e3,
        x_m[-1] * 1e3,
        x_m[0] * 1e3,
        x_m[-1] * 1e3,
    )
    amp_image = axes[0].imshow(
        amplitude_norm,
        extent=common_extent,
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axes[0].contour(
        x_m * 1e3,
        x_m * 1e3,
        amplitude_norm,
        levels=(0.5,),
        colors=("cyan",),
        linewidths=1.2,
    )
    axes[0].set_title("Pressure amplitude |p| / max; cyan = half maximum")
    fig.colorbar(amp_image, ax=axes[0], label="normalised amplitude")
    axes[0].set_xlim(-limit_mm, limit_mm)
    axes[0].set_ylim(-limit_mm, limit_mm)
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_aspect("equal")

    axes[1].plot(x_m * 1e3, x_amp_norm, label="x profile")
    axes[1].plot(x_m * 1e3, y_amp_norm, "--", label="y profile")
    axes[1].axhline(0.5, color="gray", linewidth=1)
    axes[1].axvspan(x_amp_left * 1e3, x_amp_right * 1e3, color="cyan", alpha=0.18)
    axes[1].set_title(
        f"Amplitude FWHM: {x_amp_fwhm_m * 1e3:.3f} × "
        f"{y_amp_fwhm_m * 1e3:.3f} mm"
    )
    axes[1].set_ylabel("normalised |p|")
    axes[1].legend()
    axes[1].set_xlim(-limit_mm, limit_mm)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("transverse position [mm]")

    fig.suptitle(args.title, fontsize=16)
    fig.savefig(args.image, dpi=220)
    plt.close(fig)

    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("quantity", "x_mm", "y_mm"))
        writer.writerow(("pressure_amplitude_fwhm", x_amp_fwhm_m * 1e3, y_amp_fwhm_m * 1e3))
        writer.writerow(("pressure_amplitude_half_max_left", x_amp_left * 1e3, ""))
        writer.writerow(("pressure_amplitude_half_max_right", x_amp_right * 1e3, ""))
    print(f"Saved {args.image}")
    print(f"Amplitude FWHM [mm]: {x_amp_fwhm_m * 1e3:.6f} x {y_amp_fwhm_m * 1e3:.6f}")


if __name__ == "__main__":
    main()
