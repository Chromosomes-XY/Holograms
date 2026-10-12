"""Matched Q=0 point and Q=1 ring holograms with XY/XZ intensity fields."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hologramGen import (
    acoustic_params,
    apply_hologram,
    impedance_params,
    make_circular_pzt,
    make_grid,
    propagate_ASM,
)


FREQUENCY_HZ = 2.05e6
PZT_DIAMETER_M = 0.020
FOCAL_DISTANCE_M = 0.030
DOMAIN_WIDTH_M = 0.024
N = 512
THICKNESS_MIN_M = 0.0
THICKNESS_MAX_M = 2.0e-3
N_PHASE_OFFSETS = 96
OUTPUT_DIR = Path("point_q1_intensity_comparison")


def make_thickness(base_phase, offset, phase_per_thickness, pzt):
    phase = np.mod(base_phase + offset, 2.0 * np.pi)
    thickness = THICKNESS_MIN_M + np.mod(
        phase - phase_per_thickness * THICKNESS_MIN_M,
        2.0 * np.pi,
    ) / phase_per_thickness
    if np.nanmax(thickness) > THICKNESS_MAX_M + 1e-12:
        raise RuntimeError("Mapped thickness exceeds the 2 mm allowance.")
    thickness[pzt == 0] = np.nan
    return thickness


def radial_profile(values, radius, dr):
    bins = np.arange(0.0, radius.max() + dr, dr)
    index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, len(bins) - 2)
    sums = np.bincount(index, weights=values.ravel(), minlength=len(bins) - 1)
    counts = np.bincount(index, minlength=len(bins) - 1)
    return 0.5 * (bins[:-1] + bins[1:]), sums / np.maximum(counts, 1)


def fwhm(x, y):
    peak = int(np.argmax(y))
    half = 0.5 * y[peak]
    left = peak
    right = peak
    while left > 0 and y[left - 1] >= half:
        left -= 1
    while right + 1 < y.size and y[right + 1] >= half:
        right += 1

    def crossing(i, j):
        if abs(y[j] - y[i]) < 1e-15:
            return 0.5 * (x[i] + x[j])
        return x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])

    xl = x[0] if left == 0 else crossing(left - 1, left)
    xr = x[-1] if right == y.size - 1 else crossing(right, right + 1)
    return float(xr - xl), float(x[peak])


def optimize_offset(base_phase, pzt, dx, acoustic, impedance, metric):
    phase_per_thickness = acoustic["km"] - acoustic["kh"]
    offsets = np.linspace(0.0, 2.0 * np.pi, N_PHASE_OFFSETS, endpoint=False)
    best = None
    for offset in offsets:
        thickness = make_thickness(base_phase, offset, phase_per_thickness, pzt)
        p_exit = apply_hologram(
            pzt.astype(complex), thickness,
            impedance["Zt"], impedance["Zh"], impedance["Zm"],
            acoustic["kh"], acoustic["km"],
        )
        focal = propagate_ASM(p_exit, dx, acoustic["lam_m"], FOCAL_DISTANCE_M)
        score = float(metric(np.abs(focal) ** 2))
        if best is None or score > best[0]:
            best = (score, float(offset), thickness.copy(), p_exit.copy(), focal.copy())
    return best


def point_metric(intensity):
    center = intensity.shape[0] // 2
    return intensity[center, center]


def ring_metric(intensity):
    center = intensity.shape[0] // 2
    yy, xx = np.indices(intensity.shape)
    rr_pixels = np.hypot(xx - center, yy - center)
    annulus = (rr_pixels >= 8) & (rr_pixels <= 32)
    return np.max(intensity[annulus]) - intensity[center, center]


def simulate_design(charge, label, metric, x, xv, yv, dx, pzt, acoustic, impedance):
    radius = np.hypot(xv, yv)
    azimuth = np.arctan2(yv, xv)
    path_difference = np.sqrt(radius**2 + FOCAL_DISTANCE_M**2) - FOCAL_DISTANCE_M
    base_phase = -acoustic["km"] * path_difference + charge * azimuth
    score, offset, thickness, p_exit, focal_complex = optimize_offset(
        base_phase, pzt, dx, acoustic, impedance, metric
    )
    focal_intensity = np.abs(focal_complex) ** 2
    focal_normalized = focal_intensity / (focal_intensity.max() + 1e-15)

    center = N // 2
    z_samples = np.linspace(0.0, 0.045, 181)
    xz_intensity = np.stack(
        [
            np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z))[center]) ** 2
            for z in z_samples
        ]
    )
    xz_normalized = xz_intensity / (xz_intensity.max() + 1e-15)
    peak_index = np.unravel_index(np.argmax(xz_intensity), xz_intensity.shape)
    axial_peak_z = float(z_samples[peak_index[0]])

    radius_grid = np.hypot(xv, yv)
    if charge == 0:
        width, peak_position = fwhm(x, focal_normalized[center])
        metrics = {
            "intensity_fwhm_m": width,
            "peak_radius_m": 0.0,
            "center_to_peak": 1.0,
        }
    else:
        rr, profile = radial_profile(focal_normalized, radius_grid, dx)
        width, peak_position = fwhm(rr, profile)
        metrics = {
            "intensity_fwhm_m": width,
            "peak_radius_m": peak_position,
            "center_to_peak": float(focal_normalized[center, center]),
        }

    result = {
        "label": label,
        "charge": charge,
        "offset": offset,
        "score": score,
        "thickness": thickness,
        "focal_intensity": focal_intensity,
        "focal_normalized": focal_normalized,
        "z_samples": z_samples,
        "xz_intensity": xz_intensity,
        "xz_normalized": xz_normalized,
        "axial_peak_z": axial_peak_z,
        **metrics,
    }
    return result


def plot_design(result, x, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), constrained_layout=True)
    x_mm = x * 1e3

    im = axes[0].pcolormesh(
        x_mm, x_mm, result["thickness"] * 1e3,
        shading="auto", cmap="viridis", vmin=0, vmax=THICKNESS_MAX_M * 1e3,
    )
    axes[0].set_aspect("equal")
    axes[0].set(title="Hologram thickness", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=axes[0], label="thickness [mm]")

    im = axes[1].pcolormesh(
        x_mm, x_mm, result["focal_normalized"],
        shading="auto", cmap="inferno", vmin=0, vmax=1,
    )
    axes[1].contour(
        x_mm, x_mm, result["focal_normalized"], levels=[0.5],
        colors="cyan", linewidths=1,
    )
    axes[1].set_aspect("equal")
    axes[1].set(
        xlim=(-4, 4), ylim=(-4, 4),
        title="XY intensity at z = 30 mm", xlabel="x [mm]", ylabel="y [mm]",
    )
    fig.colorbar(im, ax=axes[1], label="normalised intensity")

    im = axes[2].pcolormesh(
        result["z_samples"] * 1e3, x_mm, result["xz_normalized"].T,
        shading="auto", cmap="inferno", vmin=0, vmax=1,
    )
    axes[2].axvline(30, color="cyan", ls="--", lw=1.2, label="demanded plane")
    axes[2].axvline(
        result["axial_peak_z"] * 1e3, color="lime", ls=":", lw=1.2,
        label="axial maximum",
    )
    axes[2].set(
        xlim=(0, 45), ylim=(-6, 6),
        title="XZ intensity field", xlabel="z [mm]", ylabel="x [mm]",
    )
    axes[2].legend(loc="upper right")
    fig.colorbar(im, ax=axes[2], label="normalised intensity")

    if result["charge"] == 0:
        detail = f"intensity FWHM {result['intensity_fwhm_m']*1e3:.3f} mm"
    else:
        detail = (
            f"ring radius {result['peak_radius_m']*1e3:.3f} mm; "
            f"radial intensity FWHM {result['intensity_fwhm_m']*1e3:.3f} mm"
        )
    fig.suptitle(
        f"{result['label']} — 20 mm circular PZT, 30 mm demand; {detail}"
    )
    fig.savefig(output_path, dpi=190)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()
    phase_period = 2.0 * np.pi / (acoustic["km"] - acoustic["kh"])
    if phase_period > THICKNESS_MAX_M - THICKNESS_MIN_M:
        raise RuntimeError("The 0–2 mm thickness range cannot encode 2π.")

    point = simulate_design(
        0, "Q = 0 optimized point focus", point_metric,
        x, xv, yv, dx, pzt, acoustic, impedance,
    )
    ring = simulate_design(
        1, "Q = 1 optimized hollow ring", ring_metric,
        x, xv, yv, dx, pzt, acoustic, impedance,
    )

    for result, stem in ((point, "q0_point"), (ring, "q1_ring")):
        plot_design(result, x, OUTPUT_DIR / f"{stem}_three_panels.png")
        np.savez_compressed(
            OUTPUT_DIR / f"{stem}.npz",
            x_m=x,
            dx_m=dx,
            pzt_mask=pzt,
            thickness_m=result["thickness"],
            focal_intensity=result["focal_intensity"],
            focal_normalized_intensity=result["focal_normalized"],
            z_samples_m=result["z_samples"],
            xz_intensity=result["xz_intensity"],
            xz_normalized_intensity=result["xz_normalized"],
            optimized_global_phase_offset_rad=result["offset"],
            axial_peak_z_m=result["axial_peak_z"],
            intensity_fwhm_m=result["intensity_fwhm_m"],
            peak_radius_m=result["peak_radius_m"],
            center_to_peak_intensity=result["center_to_peak"],
            charge=result["charge"],
        )

    print(f"Full phase thickness period: {phase_period*1e3:.4f} mm")
    print(
        f"Q0: offset={point['offset']:.4f} rad, intensity FWHM="
        f"{point['intensity_fwhm_m']*1e3:.4f} mm, axial peak="
        f"{point['axial_peak_z']*1e3:.3f} mm"
    )
    print(
        f"Q1: offset={ring['offset']:.4f} rad, radius="
        f"{ring['peak_radius_m']*1e3:.4f} mm, radial intensity FWHM="
        f"{ring['intensity_fwhm_m']*1e3:.4f} mm, center/peak intensity="
        f"{ring['center_to_peak']:.6f}, axial peak={ring['axial_peak_z']*1e3:.3f} mm"
    )


if __name__ == "__main__":
    main()
