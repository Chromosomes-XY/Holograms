"""Lowest-order analytic focused-vortex lens for a fine hollow ring."""

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
VORTEX_CHARGE = 1
DOMAIN_WIDTH_M = 0.024
N = 512
THICKNESS_MIN_M = 0.0
THICKNESS_MAX_M = 2.0e-3
OUTPUT_DIR = Path("analytic_vortex_ring_30mm_20mm_pzt")


def radial_profile(amplitude, xv, yv, dr):
    radius = np.hypot(xv, yv)
    bins = np.arange(0.0, radius.max() + dr, dr)
    index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, len(bins) - 2)
    sums = np.bincount(index, weights=amplitude.ravel(), minlength=len(bins) - 1)
    counts = np.bincount(index, minlength=len(bins) - 1)
    return 0.5 * (bins[:-1] + bins[1:]), sums / np.maximum(counts, 1)


def peak_fwhm(x, y):
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


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    radius = np.hypot(xv, yv)
    azimuth = np.arctan2(yv, xv)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()

    path_difference = np.sqrt(radius**2 + FOCAL_DISTANCE_M**2) - FOCAL_DISTANCE_M
    focusing_phase = -acoustic["km"] * path_difference
    total_phase = np.mod(focusing_phase + VORTEX_CHARGE * azimuth, 2.0 * np.pi)

    phase_per_thickness = acoustic["km"] - acoustic["kh"]
    phase_period_thickness = 2.0 * np.pi / phase_per_thickness
    if phase_period_thickness > THICKNESS_MAX_M - THICKNESS_MIN_M:
        raise RuntimeError("The 0–2 mm thickness range cannot represent a full phase cycle.")
    thickness = THICKNESS_MIN_M + np.mod(
        total_phase - phase_per_thickness * THICKNESS_MIN_M,
        2.0 * np.pi,
    ) / phase_per_thickness
    thickness[~pzt.astype(bool)] = np.nan

    p_exit = apply_hologram(
        pzt.astype(complex),
        thickness,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )

    focal = np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], FOCAL_DISTANCE_M))
    focal_normalized = focal / (focal.max() + 1e-15)
    rr, radial = radial_profile(focal, xv, yv, dx)
    radial_normalized = radial / (radial.max() + 1e-15)
    radial_fwhm, peak_radius = peak_fwhm(rr, radial)
    center = N // 2
    center_ratio = float(focal[center, center] / (focal.max() + 1e-15))

    z_samples = np.linspace(0.0, 0.045, 181)
    longitudinal = np.stack(
        [
            np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z)))[center]
            for z in z_samples
        ]
    )
    longitudinal_normalized = longitudinal / (longitudinal.max() + 1e-15)

    np.savez_compressed(
        OUTPUT_DIR / "analytic_vortex_ring_30mm_20mm_pzt.npz",
        x_m=x,
        dx_m=dx,
        pzt_mask=pzt,
        total_phase_rad=total_phase,
        thickness_m=thickness,
        focal_amplitude=focal,
        z_samples_m=z_samples,
        longitudinal_amplitude=longitudinal,
        radial_r_m=rr,
        radial_amplitude=radial,
        peak_radius_m=peak_radius,
        radial_fwhm_m=radial_fwhm,
        center_to_peak_ratio=center_ratio,
        vortex_charge=VORTEX_CHARGE,
    )

    fig = plt.figure(figsize=(16.5, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.15, 1.3))
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_thickness = fig.add_subplot(gs[1, 0])
    ax_plane = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, 1])
    ax_long = fig.add_subplot(gs[:, 2])
    x_mm = x * 1e3

    im = ax_phase.pcolormesh(
        x_mm, x_mm, np.where(pzt, total_phase, np.nan), shading="auto",
        cmap="viridis", vmin=0, vmax=2 * np.pi,
    )
    ax_phase.set_aspect("equal")
    ax_phase.set(title="Single Q = 1 focused-vortex phase", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_phase, label="phase [rad]", ticks=[0, np.pi, 2 * np.pi])

    im = ax_thickness.pcolormesh(
        x_mm, x_mm, thickness * 1e3, shading="auto",
        cmap="viridis", vmin=0, vmax=THICKNESS_MAX_M * 1e3,
    )
    ax_thickness.set_aspect("equal")
    ax_thickness.set(title="Mapped hologram thickness", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_thickness, label="thickness [mm]")

    im = ax_plane.pcolormesh(
        x_mm, x_mm, focal_normalized, shading="auto", cmap="inferno", vmin=0, vmax=1,
    )
    ax_plane.contour(x_mm, x_mm, focal_normalized, levels=[0.5], colors="cyan", linewidths=1)
    ax_plane.set_aspect("equal")
    ax_plane.set(xlim=(-4, 4), ylim=(-4, 4), title="Fine ring at z = 30 mm", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_plane, label="plane-normalised |p|")

    ax_profile.plot(rr * 1e3, radial_normalized, color="tab:orange", lw=2)
    ax_profile.axhline(0.5, color="0.4", ls="--", lw=1)
    ax_profile.axvline(peak_radius * 1e3, color="deepskyblue", ls=":", lw=1.2)
    ax_profile.set(
        xlim=(0, 4), ylim=(0, 1.05), xlabel="radius [mm]", ylabel="normalised |p|",
        title=f"Peak radius {peak_radius*1e3:.3f} mm; radial FWHM {radial_fwhm*1e3:.3f} mm",
    )
    ax_profile.grid(alpha=0.25)

    im = ax_long.pcolormesh(
        z_samples * 1e3, x_mm, longitudinal_normalized.T, shading="auto",
        cmap="inferno", vmin=0, vmax=1,
    )
    ax_long.axvline(30, color="cyan", ls="--", lw=1.2)
    ax_long.set(xlim=(0, 45), ylim=(-6, 6), title="Longitudinal ring amplitude |p|", xlabel="z [mm]", ylabel="x [mm]")
    fig.colorbar(im, ax=ax_long, label="global normalised amplitude")

    fig.suptitle("Fine hollow ring from one Q = 1 analytic phase — 20 mm PZT, 30 mm focus")
    fig.savefig(OUTPUT_DIR / "analytic_vortex_ring_30mm_20mm_pzt.png", dpi=190)
    plt.close(fig)

    print(f"Actual thickness range: {np.nanmin(thickness)*1e3:.4f}–{np.nanmax(thickness)*1e3:.4f} mm")
    print(f"Peak ring radius: {peak_radius*1e3:.4f} mm")
    print(f"Radial amplitude FWHM: {radial_fwhm*1e3:.4f} mm")
    print(f"On-axis / ring-peak amplitude: {center_ratio:.5f}")


if __name__ == "__main__":
    main()
