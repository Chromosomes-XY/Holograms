"""Design a 20 mm circular-PZT hologram for a 1 mm-radius ring at z=30 mm."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hologramGen import (
    acoustic_params,
    apply_hologram,
    design_hologram_multi_plane,
    impedance_params,
    make_circular_pzt,
    make_grid,
    make_ring_target,
    propagate_ASM,
)


FREQUENCY_HZ = 2.05e6
PZT_DIAMETER_M = 0.020
DOMAIN_WIDTH_M = 0.024
N = 512
FOCAL_DISTANCE_M = 0.030
RING_RADIUS_M = 0.001
N_ITER = 80
TMIN_M = 0.4e-3
TMAX_M = 4.0e-3
OUTPUT_DIR = Path("ring_focus_30mm_20mm_pzt")


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

    def cross(i, j):
        if abs(y[j] - y[i]) < 1e-15:
            return 0.5 * (x[i] + x[j])
        return x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])

    xl = x[0] if left == 0 else cross(left - 1, left)
    xr = x[-1] if right == y.size - 1 else cross(right, right + 1)
    return float(xr - xl), float(x[peak])


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()

    # The requested radial forcing thickness is the finest resolvable cell.
    target = make_ring_target(xv, yv, RING_RADIUS_M, dx)
    thickness, history = design_hologram_multi_plane(
        pzt,
        target[None, ...],
        np.array([FOCAL_DISTANCE_M]),
        dx,
        acoustic["lam_m"],
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
        n_iter=N_ITER,
        Tmin=TMIN_M,
        Tmax=TMAX_M,
        vortex_charge=1,
    )

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

    z_samples = np.linspace(0.0, 0.045, 181)
    center = N // 2
    longitudinal = np.stack(
        [
            np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z)))[center]
            for z in z_samples
        ]
    )
    longitudinal /= longitudinal.max() + 1e-15

    rr, radial = radial_profile(focal, xv, yv, dx)
    radial_width, measured_radius = peak_fwhm(rr, radial)
    radial_normalized = radial / (radial.max() + 1e-15)

    np.savez_compressed(
        OUTPUT_DIR / "ring_focus_30mm_20mm_pzt.npz",
        x_m=x,
        dx_m=dx,
        pzt_mask=pzt,
        target_amplitude=target,
        thickness_m=thickness,
        focal_amplitude=focal,
        z_samples_m=z_samples,
        longitudinal_amplitude=longitudinal,
        radial_r_m=rr,
        radial_amplitude=radial,
        measured_peak_radius_m=measured_radius,
        radial_fwhm_m=radial_width,
        vortex_charge=1,
        history_corr=np.asarray(history["amp_corr"]),
    )

    fig = plt.figure(figsize=(15.5, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.25, 1.0, 1.0))
    ax_long, ax_plane, ax_profile = [fig.add_subplot(gs[0, i]) for i in range(3)]

    x_mm = x * 1e3
    im = ax_long.pcolormesh(
        z_samples * 1e3,
        x_mm,
        longitudinal.T,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    ax_long.axvline(30.0, color="cyan", ls="--", lw=1.2)
    ax_long.scatter([30.0, 30.0], [-1.0, 1.0], facecolors="none", edgecolors="cyan", s=28)
    ax_long.set(xlim=(0, 45), ylim=(-6, 6), xlabel="z [mm]", ylabel="x [mm]")
    ax_long.set_title("Longitudinal amplitude |p|")
    fig.colorbar(im, ax=ax_long, label="global normalised amplitude")

    focal_normalized = focal / (focal.max() + 1e-15)
    im = ax_plane.pcolormesh(x_mm, x_mm, focal_normalized, shading="auto", cmap="inferno", vmin=0, vmax=1)
    ax_plane.contour(x_mm, x_mm, focal_normalized, levels=[0.5], colors="cyan", linewidths=1.0)
    demanded = plt.Circle((0, 0), RING_RADIUS_M * 1e3, fill=False, color="deepskyblue", ls="--", lw=1.1)
    ax_plane.add_patch(demanded)
    ax_plane.set(xlim=(-4, 4), ylim=(-4, 4), xlabel="x [mm]", ylabel="y [mm]")
    ax_plane.set_aspect("equal")
    ax_plane.set_title("Focal plane at z = 30 mm")
    fig.colorbar(im, ax=ax_plane, label="plane-normalised |p|")

    ax_profile.plot(rr * 1e3, radial_normalized, color="tab:orange", lw=2, label="simulated")
    ax_profile.axvline(1.0, color="deepskyblue", ls="--", lw=1.2, label="demanded radius")
    ax_profile.axhline(0.5, color="0.4", ls=":", lw=1)
    ax_profile.set(xlim=(0, 4), ylim=(0, 1.05), xlabel="radius [mm]", ylabel="normalised |p|")
    ax_profile.set_title(
        f"Radial amplitude profile\npeak {measured_radius*1e3:.3f} mm; FWHM {radial_width*1e3:.3f} mm"
    )
    ax_profile.legend()
    ax_profile.grid(alpha=0.25)

    fig.suptitle("1 mm-radius ring focus — 20 mm circular PZT, z = 30 mm")
    fig.savefig(OUTPUT_DIR / "ring_focus_30mm_20mm_pzt_fields.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.7, 5.7), constrained_layout=True)
    im = ax.pcolormesh(x_mm, x_mm, np.where(np.isfinite(thickness), thickness * 1e3, np.nan), shading="auto")
    ax.set_aspect("equal")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title="Hologram thickness on 20 mm circular PZT")
    fig.colorbar(im, ax=ax, label="thickness [mm]")
    fig.savefig(OUTPUT_DIR / "ring_focus_30mm_20mm_pzt_hologram.png", dpi=190)
    plt.close(fig)

    print(f"Grid cell / target FWHM: {dx*1e3:.4f} mm")
    print(f"Measured peak radius: {measured_radius*1e3:.4f} mm")
    print(f"Measured radial amplitude FWHM: {radial_width*1e3:.4f} mm")
    print(f"Final target correlation: {history['amp_corr'][-1]:.5f}")
    print(
        f"Actual hologram thickness: {np.nanmin(thickness)*1e3:.4f} to "
        f"{np.nanmax(thickness)*1e3:.4f} mm"
    )


if __name__ == "__main__":
    main()
