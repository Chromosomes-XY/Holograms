"""Analytic uncharged focusing lens from phase map (a) of the reference figure."""

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
OUTPUT_DIR = Path("analytic_lens_30mm_20mm_pzt")


def fwhm_centered(x, y):
    y = np.asarray(y, dtype=float)
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
    return float(xr - xl)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    radius = np.hypot(xv, yv)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()

    # With the exp(+ikz) ASM convention, a converging spherical wave requires
    # negative path-compensation phase relative to the aperture centre.
    optical_path_difference = np.sqrt(radius**2 + FOCAL_DISTANCE_M**2) - FOCAL_DISTANCE_M
    demanded_phase = np.mod(-acoustic["km"] * optical_path_difference, 2.0 * np.pi)

    phase_per_thickness = acoustic["km"] - acoustic["kh"]
    phase_period_thickness = 2.0 * np.pi / phase_per_thickness
    if phase_period_thickness > THICKNESS_MAX_M - THICKNESS_MIN_M:
        raise RuntimeError("The allowed thickness range cannot encode a full 2π phase cycle.")

    # Offset-aware mapping preserves the complete 2π phase range. No clipping
    # occurs because the 1.777 mm modulation fits inside the new 0–2 mm limit.
    thickness = THICKNESS_MIN_M + np.mod(
        demanded_phase - phase_per_thickness * THICKNESS_MIN_M,
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

    z_samples = np.linspace(0.0, 0.045, 181)
    center = N // 2
    longitudinal = np.stack(
        [
            np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z)))[center]
            for z in z_samples
        ]
    )
    global_peak_index = np.unravel_index(np.argmax(longitudinal), longitudinal.shape)
    measured_peak_z = float(z_samples[global_peak_index[0]])
    longitudinal_normalized = longitudinal / (longitudinal.max() + 1e-15)

    focal = np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], FOCAL_DISTANCE_M))
    focal_normalized = focal / (focal.max() + 1e-15)
    focal_cut = focal_normalized[center]
    amplitude_fwhm = fwhm_centered(x, focal_cut)

    np.savez_compressed(
        OUTPUT_DIR / "analytic_lens_30mm_20mm_pzt.npz",
        x_m=x,
        dx_m=dx,
        pzt_mask=pzt,
        demanded_phase_rad=demanded_phase,
        thickness_m=thickness,
        phase_period_thickness_m=phase_period_thickness,
        z_samples_m=z_samples,
        longitudinal_amplitude=longitudinal,
        focal_amplitude=focal,
        amplitude_fwhm_m=amplitude_fwhm,
        measured_peak_z_m=measured_peak_z,
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
        x_mm,
        x_mm,
        np.where(pzt, demanded_phase, np.nan),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=2 * np.pi,
    )
    ax_phase.set_aspect("equal")
    ax_phase.set(title="Concentric phase map (figure a)", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_phase, label="phase [rad]", ticks=[0, np.pi, 2 * np.pi])

    im = ax_thickness.pcolormesh(
        x_mm,
        x_mm,
        thickness * 1e3,
        shading="auto",
        cmap="viridis",
        vmin=THICKNESS_MIN_M * 1e3,
        vmax=THICKNESS_MAX_M * 1e3,
    )
    ax_thickness.set_aspect("equal")
    ax_thickness.set(title="Mapped hologram thickness", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_thickness, label="thickness [mm]")

    im = ax_plane.pcolormesh(
        x_mm,
        x_mm,
        focal_normalized,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ax_plane.contour(x_mm, x_mm, focal_normalized, levels=[0.5], colors="cyan", linewidths=1)
    ax_plane.set_aspect("equal")
    ax_plane.set(xlim=(-4, 4), ylim=(-4, 4), title="Amplitude at demanded z = 30 mm", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=ax_plane, label="plane-normalised |p|")

    ax_profile.plot(x_mm, focal_cut, color="tab:orange", lw=2)
    ax_profile.axhline(0.5, color="0.4", ls="--", lw=1)
    ax_profile.set(
        xlim=(-4, 4),
        ylim=(0, 1.05),
        xlabel="x [mm]",
        ylabel="normalised |p|",
        title=f"Focal-plane amplitude FWHM = {amplitude_fwhm*1e3:.3f} mm",
    )
    ax_profile.grid(alpha=0.25)

    im = ax_long.pcolormesh(
        z_samples * 1e3,
        x_mm,
        longitudinal_normalized.T,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ax_long.axvline(30, color="cyan", ls="--", lw=1.2, label="demanded plane")
    ax_long.axvline(measured_peak_z * 1e3, color="lime", ls=":", lw=1.2, label="axial maximum")
    ax_long.set(xlim=(0, 45), ylim=(-6, 6), title="Longitudinal amplitude |p|", xlabel="z [mm]", ylabel="x [mm]")
    ax_long.legend(loc="upper right")
    fig.colorbar(im, ax=ax_long, label="global normalised amplitude")

    fig.suptitle("Uncharged analytic focusing lens — 20 mm PZT, demanded focus 30 mm")
    fig.savefig(OUTPUT_DIR / "analytic_lens_30mm_20mm_pzt.png", dpi=190)
    plt.close(fig)

    print(f"Full 2π thickness period: {phase_period_thickness*1e3:.4f} mm")
    print(f"Actual thickness range: {np.nanmin(thickness)*1e3:.4f}–{np.nanmax(thickness)*1e3:.4f} mm")
    print(f"Axial maximum: z={measured_peak_z*1e3:.3f} mm")
    print(f"Amplitude FWHM at z=30 mm: {amplitude_fwhm*1e3:.4f} mm")


if __name__ == "__main__":
    main()
