"""Design a 200 mm circular-PZT hologram for a contracting conical field.

The constrained surface begins as a 20/3 mm diameter circle at z=20 mm,
contracts linearly to a point at (0, 0, 30 mm), and expands symmetrically to
the same circle at z=40 mm. Horizontal slices are represented by the narrowest
Gaussian ring supported by the grid, with the strongest constraint at the apex.
"""

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
PZT_DIAMETER_M = 0.200
DOMAIN_WIDTH_M = 0.220
N = 768
Z_START_M = 0.020
Z_APEX_M = 0.030
Z_END_M = 0.040
START_DIAMETER_M = 0.020 / 3.0
N_TARGET_PLANES = 31
N_ITER = 50
TMIN_M = 0.4e-3
TMAX_M = 4.0e-3
OUTPUT_DIR = Path("cone_focus_30mm_200mm_pzt")


def radial_profile(amplitude, xv, yv, dr):
    radius = np.hypot(xv, yv)
    bins = np.arange(0.0, radius.max() + dr, dr)
    index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, len(bins) - 2)
    sums = np.bincount(index, weights=amplitude.ravel(), minlength=len(bins) - 1)
    counts = np.bincount(index, minlength=len(bins) - 1)
    profile = sums / np.maximum(counts, 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, profile


def half_max_width(x, profile):
    profile = np.asarray(profile, dtype=float)
    peak_index = int(np.argmax(profile))
    half = 0.5 * profile[peak_index]
    above = profile >= half
    left = peak_index
    right = peak_index
    while left > 0 and above[left - 1]:
        left -= 1
    while right + 1 < profile.size and above[right + 1]:
        right += 1

    def crossing(i0, i1):
        x0, x1 = x[i0], x[i1]
        y0, y1 = profile[i0], profile[i1]
        if abs(y1 - y0) < 1e-15:
            return 0.5 * (x0 + x1)
        return x0 + (half - y0) * (x1 - x0) / (y1 - y0)

    xl = x[0] if left == 0 else crossing(left - 1, left)
    xr = x[-1] if right == profile.size - 1 else crossing(right, right + 1)
    return float(xr - xl)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()

    target_z = np.linspace(Z_START_M, Z_END_M, N_TARGET_PLANES)
    start_radius = 0.5 * START_DIAMETER_M
    target_radii = start_radius * np.abs(target_z - Z_APEX_M) / (
        Z_APEX_M - Z_START_M
    )

    # Symmetric raised-cosine weighting: 4x stronger at the focal plane than
    # at either boundary circle, while retaining the complete double cone.
    normalized_offset = np.abs(target_z - Z_APEX_M) / (Z_END_M - Z_APEX_M)
    plane_weights = 0.25 + 0.75 * np.cos(0.5 * np.pi * normalized_offset) ** 2

    # One grid cell is the narrowest meaningful requested surface thickness.
    raw_targets = np.stack(
        [make_ring_target(xv, yv, radius, dx) for radius in target_radii]
    )
    # Equal L2 energy per slice represents flux concentrating as the cone closes.
    target_norms = np.sqrt(np.sum(raw_targets**2, axis=(1, 2), keepdims=True))
    targets = raw_targets / np.maximum(target_norms, 1e-15)

    print(
        f"Grid: {N}x{N}, dx={dx*1e3:.4f} mm, "
        f"lambda={acoustic['lam_m']*1e3:.4f} mm"
    )
    print(
        f"Cone: diameter={START_DIAMETER_M*1e3:.4f} mm at 20 mm, "
        "apex=(0,0,30 mm), symmetric exit circle at 40 mm"
    )

    thickness, history = design_hologram_multi_plane(
        pzt,
        targets,
        target_z,
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
        plane_weights=plane_weights,
        enforce_uniform_target_phase=True,
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

    z_samples = np.linspace(0.0, 0.050, 201)
    center = N // 2
    longitudinal = np.empty((z_samples.size, N), dtype=np.float32)
    for i, z in enumerate(z_samples):
        longitudinal[i] = np.abs(
            propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z))[center]
        )
    longitudinal /= longitudinal.max() + 1e-15

    plane_z = np.array([Z_START_M, 0.025, Z_APEX_M, 0.035, Z_END_M])
    plane_fields = []
    for z in plane_z:
        field = np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z)))
        plane_fields.append(field)
    plane_fields = np.stack(plane_fields)

    radial_r, focal_radial = radial_profile(plane_fields[2], xv, yv, dx)
    focal_fwhm = half_max_width(radial_r, focal_radial)

    measured_peak_radii = []
    for z in target_z:
        field = np.abs(propagate_ASM(p_exit, dx, acoustic["lam_m"], float(z)))
        rr, pp = radial_profile(field, xv, yv, dx)
        local = rr <= 0.008
        measured_peak_radii.append(float(rr[local][np.argmax(pp[local])]))
    measured_peak_radii = np.asarray(measured_peak_radii)

    np.savez_compressed(
        OUTPUT_DIR / "cone_focus_30mm_200mm_pzt.npz",
        x_m=x,
        dx_m=dx,
        thickness_m=thickness,
        pzt_mask=pzt,
        target_z_m=target_z,
        target_radii_m=target_radii,
        plane_weights=plane_weights,
        measured_peak_radii_m=measured_peak_radii,
        z_samples_m=z_samples,
        longitudinal_amplitude=longitudinal,
        display_plane_z_m=plane_z,
        display_plane_amplitude=plane_fields,
        focal_radial_r_m=radial_r,
        focal_radial_amplitude=focal_radial,
        focal_fwhm_m=focal_fwhm,
        history_corr=np.asarray(history["amp_corr"]),
    )

    fig = plt.figure(figsize=(17, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, width_ratios=(1.3, 1.0, 1.0))
    ax_long = fig.add_subplot(gs[:, 0])
    ax_20 = fig.add_subplot(gs[0, 1])
    ax_25 = fig.add_subplot(gs[0, 2])
    ax_30 = fig.add_subplot(gs[1, 1])
    ax_35 = fig.add_subplot(gs[1, 2])
    ax_40 = fig.add_subplot(gs[2, 1])
    ax_prof = fig.add_subplot(gs[2, 2])

    x_mm = x * 1e3
    image = ax_long.pcolormesh(
        z_samples * 1e3,
        x_mm,
        longitudinal.T,
        shading="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax_long.plot(target_z * 1e3, target_radii * 1e3, "c--", lw=1.5)
    ax_long.plot(target_z * 1e3, -target_radii * 1e3, "c--", lw=1.5)
    ax_long.scatter([30], [0], s=28, facecolors="none", edgecolors="cyan")
    ax_long.set(xlim=(0, 50), ylim=(-10, 10), xlabel="z [mm]", ylabel="x [mm]")
    ax_long.set_title("Longitudinal amplitude |p|\ncyan: demanded symmetric surface")
    fig.colorbar(image, ax=ax_long, label="global normalised amplitude")

    limits = [6.0, 5.0, 3.0, 5.0, 6.0]
    titles = [
        "Entrance circle: z = 20 mm",
        "Contracting: z = 25 mm",
        "Weighted apex: z = 30 mm",
        "Expanding: z = 35 mm",
        "Exit circle: z = 40 mm",
    ]
    for ax, field, limit, title in zip(
        (ax_20, ax_25, ax_30, ax_35, ax_40), plane_fields, limits, titles
    ):
        norm = field / (field.max() + 1e-15)
        im = ax.pcolormesh(x_mm, x_mm, norm, shading="auto", cmap="inferno", vmin=0, vmax=1)
        ax.contour(x_mm, x_mm, norm, levels=[0.5], colors="cyan", linewidths=0.9)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x [mm]", ylabel="y [mm]")
        ax.set_aspect("equal")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="plane-normalised |p|")

    focal_norm = focal_radial / (focal_radial.max() + 1e-15)
    ax_prof.plot(radial_r * 1e3, focal_norm, color="tab:orange", lw=2)
    ax_prof.axhline(0.5, color="0.4", ls="--", lw=1)
    ax_prof.set(xlim=(0, 3), ylim=(0, 1.05), xlabel="radius [mm]", ylabel="normalised |p|")
    ax_prof.set_title(f"Apex radial profile\nradial half-width = {focal_fwhm*1e3:.3f} mm")
    ax_prof.grid(alpha=0.25)

    fig.suptitle(
        "Symmetric double-cone boundary — 200 mm circular PZT, weighted apex at 30 mm",
        fontsize=15,
    )
    fig.savefig(OUTPUT_DIR / "cone_focus_30mm_200mm_pzt_fields.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    finite = np.isfinite(thickness)
    im = ax.pcolormesh(x_mm, x_mm, np.where(finite, thickness * 1e3, np.nan), shading="auto")
    ax.set_aspect("equal")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title="Hologram thickness on 200 mm circular PZT")
    fig.colorbar(im, ax=ax, label="thickness [mm]")
    fig.savefig(OUTPUT_DIR / "cone_focus_30mm_200mm_pzt_hologram.png", dpi=180)
    plt.close(fig)

    print(f"Final mean plane correlation: {history['amp_corr'][-1]:.5f}")
    print(f"Apex radial half-width at half maximum: {focal_fwhm*1e3:.4f} mm")
    print(
        f"Actual hologram thickness: {np.nanmin(thickness)*1e3:.4f} to "
        f"{np.nanmax(thickness)*1e3:.4f} mm"
    )
    print(f"Saved results to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
