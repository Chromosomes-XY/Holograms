"""Compare intensity-forced point and Q=1 ring holograms at z=25 mm."""

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


FREQUENCY_HZ = 2.164e6
PZT_DIAMETER_M = 0.020
FOCAL_DISTANCE_M = 0.025
RING_RADIUS_M = 0.001
DOMAIN_WIDTH_M = 0.024
N = 512
N_ITER = 100
THICKNESS_MIN_M = 0.0
THICKNESS_MAX_M = 2.0e-3
OUTPUT_DIR = Path("intensity_forced_point_and_q1_f25mm_2164khz")


def propagate_padded(field, dx, wavelength, distance, pad_factor=2):
    """Zero-padded ASM propagation to suppress periodic wrap-around."""
    n = field.shape[0]
    padded_n = pad_factor * n
    before = (padded_n - n) // 2
    after = padded_n - n - before
    padded = np.pad(field, ((before, after), (before, after)))
    propagated = propagate_ASM(padded, dx, wavelength, distance)
    return propagated[before : before + n, before : before + n]


def make_point_intensity(radius, fwhm):
    return np.exp(-4.0 * np.log(2.0) * radius**2 / fwhm**2)


def make_ring_intensity(radius, ring_radius, fwhm):
    return np.exp(-4.0 * np.log(2.0) * (radius - ring_radius) ** 2 / fwhm**2)


def analytic_q1_thickness(radius, azimuth, pzt, acoustic):
    """Focused spherical phase plus one azimuthal 2π winding."""
    path_difference = (
        np.sqrt(radius**2 + FOCAL_DISTANCE_M**2) - FOCAL_DISTANCE_M
    )
    phase = np.mod(-acoustic["km"] * path_difference + azimuth, 2.0 * np.pi)
    d_k = acoustic["km"] - acoustic["kh"]
    thickness = THICKNESS_MIN_M + np.mod(
        phase - d_k * THICKNESS_MIN_M,
        2.0 * np.pi,
    ) / d_k
    thickness[pzt <= 0] = np.nan
    return thickness


def design_from_intensity(
    pzt,
    target_intensity,
    target_phase_mode,
    azimuth,
    dx,
    acoustic,
    impedance,
):
    """Phase-only GS projection whose prescribed quantity is intensity."""
    actuator = pzt > 0
    p_in = pzt.astype(np.complex128)
    target_amplitude = np.sqrt(np.maximum(target_intensity, 0.0))
    d_k = acoustic["km"] - acoustic["kh"]
    eps = 1e-15

    thickness = np.full_like(pzt, np.nan, dtype=float)
    if target_phase_mode == "q1":
        initial_phase = np.mod(azimuth + np.pi, 2.0 * np.pi)
        thickness[actuator] = initial_phase[actuator] / d_k
    else:
        thickness[actuator] = THICKNESS_MIN_M

    history = []
    vortex_basis = np.exp(1j * azimuth)
    for _ in range(N_ITER):
        exit_field = apply_hologram(
            p_in,
            thickness,
            impedance["Zt"],
            impedance["Zh"],
            impedance["Zm"],
            acoustic["kh"],
            acoustic["km"],
        )
        focal_field = propagate_padded(
            exit_field,
            dx,
            acoustic["lam_m"],
            FOCAL_DISTANCE_M,
        )
        simulated_intensity = np.abs(focal_field) ** 2
        sim_norm = simulated_intensity / (simulated_intensity.max() + eps)
        tgt_norm = target_intensity / (target_intensity.max() + eps)
        history.append(float(np.mean((sim_norm - tgt_norm) ** 2)))

        if target_phase_mode == "q1":
            target_mode = target_amplitude * vortex_basis
            global_phase = np.angle(np.vdot(target_mode, focal_field))
            constrained = target_mode * np.exp(1j * global_phase)
        else:
            constrained = target_amplitude * np.exp(1j * np.angle(focal_field))

        required_exit = propagate_padded(
            constrained,
            dx,
            acoustic["lam_m"],
            -FOCAL_DISTANCE_M,
        )
        required_phase = np.angle(required_exit / (p_in + eps))
        mapped = THICKNESS_MIN_M + np.mod(
            required_phase - d_k * THICKNESS_MIN_M,
            2.0 * np.pi,
        ) / d_k
        thickness[actuator] = np.clip(
            mapped[actuator],
            THICKNESS_MIN_M,
            THICKNESS_MAX_M,
        )

    return thickness, np.asarray(history)


def fwhm_1d(x, profile):
    peak = int(np.argmax(profile))
    half = 0.5 * profile[peak]
    left = peak
    right = peak
    while left > 0 and profile[left - 1] >= half:
        left -= 1
    while right + 1 < profile.size and profile[right + 1] >= half:
        right += 1

    def cross(i, j):
        if abs(profile[j] - profile[i]) < 1e-15:
            return 0.5 * (x[i] + x[j])
        return x[i] + (half - profile[i]) * (x[j] - x[i]) / (
            profile[j] - profile[i]
        )

    xl = x[0] if left == 0 else cross(left - 1, left)
    xr = x[-1] if right == profile.size - 1 else cross(right, right + 1)
    return float(xr - xl), float(x[peak])


def radial_profile(values, radius, dr):
    bins = np.arange(0.0, radius.max() + dr, dr)
    index = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, len(bins) - 2)
    sums = np.bincount(index, weights=values.ravel(), minlength=len(bins) - 1)
    counts = np.bincount(index, minlength=len(bins) - 1)
    return 0.5 * (bins[:-1] + bins[1:]), sums / np.maximum(counts, 1)


def evaluate(thickness, pzt, x, dx, acoustic, impedance):
    p_exit = apply_hologram(
        pzt.astype(complex),
        thickness,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )
    focal_field = propagate_padded(p_exit, dx, acoustic["lam_m"], FOCAL_DISTANCE_M)
    focal_intensity = np.abs(focal_field) ** 2
    z_samples = np.linspace(0.0, 0.045, 181)
    center = len(x) // 2
    longitudinal_intensity = np.stack(
        [
            np.abs(propagate_padded(p_exit, dx, acoustic["lam_m"], float(z))[center])
            ** 2
            for z in z_samples
        ]
    )
    return focal_intensity, z_samples, longitudinal_intensity


def plot_set(name, title, thickness, focal_intensity, z_samples, longitudinal, x, metric):
    x_mm = x * 1e3
    focal_normalized = focal_intensity / (focal_intensity.max() + 1e-15)
    longitudinal_normalized = longitudinal / (longitudinal.max() + 1e-15)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)

    im = axes[0].pcolormesh(
        x_mm,
        x_mm,
        thickness * 1e3,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=THICKNESS_MAX_M * 1e3,
    )
    axes[0].set_aspect("equal")
    axes[0].set(title="Hologram thickness", xlabel="x [mm]", ylabel="y [mm]")
    fig.colorbar(im, ax=axes[0], label="thickness [mm]")

    im = axes[1].pcolormesh(
        x_mm,
        x_mm,
        focal_normalized,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axes[1].contour(x_mm, x_mm, focal_normalized, levels=[0.5], colors="cyan", linewidths=1)
    axes[1].set_aspect("equal")
    axes[1].set(
        xlim=(-4, 4),
        ylim=(-4, 4),
        title=f"Normalised x-y intensity at z = {FOCAL_DISTANCE_M*1e3:.0f} mm\n{metric}",
        xlabel="x [mm]",
        ylabel="y [mm]",
    )
    fig.colorbar(im, ax=axes[1], label="normalised intensity")

    im = axes[2].pcolormesh(
        z_samples * 1e3,
        x_mm,
        longitudinal_normalized.T,
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axes[2].axvline(FOCAL_DISTANCE_M * 1e3, color="cyan", ls="--", lw=1.2)
    axes[2].set(
        xlim=(0, 45),
        ylim=(-6, 6),
        title="Normalised x-z intensity",
        xlabel="z [mm]",
        ylabel="x [mm]",
    )
    fig.colorbar(im, ax=axes[2], label="normalised intensity")
    fig.suptitle(title)
    fig.savefig(OUTPUT_DIR / f"{name}_three_panel.png", dpi=190)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    x, xv, yv, dx = make_grid(N, DOMAIN_WIDTH_M)
    radius = np.hypot(xv, yv)
    azimuth = np.arctan2(yv, xv)
    pzt = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ)
    impedance = impedance_params()

    point_target = make_point_intensity(radius, dx)
    print("Optimising point intensity target...")
    point_thickness, point_history = design_from_intensity(
        pzt, point_target, "point", azimuth, dx, acoustic, impedance
    )
    point_focal, point_z, point_long = evaluate(
        point_thickness, pzt, x, dx, acoustic, impedance
    )

    print("Generating analytic Q=1 focused-vortex phase...")
    ring_thickness = analytic_q1_thickness(radius, azimuth, pzt, acoustic)
    ring_focal, ring_z, ring_long = evaluate(
        ring_thickness, pzt, x, dx, acoustic, impedance
    )

    center = N // 2
    point_line = point_focal[center] / (point_focal.max() + 1e-15)
    point_fwhm, _ = fwhm_1d(x, point_line)

    rr, ring_radial = radial_profile(ring_focal, radius, dx)
    ring_radial /= ring_radial.max() + 1e-15
    ring_fwhm, ring_peak_radius = fwhm_1d(rr, ring_radial)
    ring_center_ratio = float(ring_focal[center, center] / (ring_focal.max() + 1e-15))

    plot_set(
        "point",
        "Intensity-forced point focus — 20 mm PZT, 2.164 MHz, z = 25 mm",
        point_thickness,
        point_focal,
        point_z,
        point_long,
        x,
        f"intensity FWHM = {point_fwhm*1e3:.3f} mm",
    )
    plot_set(
        "q1_ring",
        "Analytic phase-field Q = 1 ring — 20 mm PZT, 2.164 MHz, z = 25 mm",
        ring_thickness,
        ring_focal,
        ring_z,
        ring_long,
        x,
        f"radius = {ring_peak_radius*1e3:.3f} mm; radial FWHM = {ring_fwhm*1e3:.3f} mm",
    )

    np.savez_compressed(
        OUTPUT_DIR / "intensity_forced_point_and_q1_f25mm_2164khz.npz",
        x_m=x,
        dx_m=dx,
        point_target_intensity=point_target,
        point_thickness_m=point_thickness,
        point_focal_intensity=point_focal,
        point_z_samples_m=point_z,
        point_longitudinal_intensity=point_long,
        point_history_mse=point_history,
        point_intensity_fwhm_m=point_fwhm,
        ring_thickness_m=ring_thickness,
        ring_focal_intensity=ring_focal,
        ring_z_samples_m=ring_z,
        ring_longitudinal_intensity=ring_long,
        ring_peak_radius_m=ring_peak_radius,
        ring_radial_intensity_fwhm_m=ring_fwhm,
        ring_center_to_peak_intensity=ring_center_ratio,
    )

    print(f"Point intensity FWHM: {point_fwhm*1e3:.4f} mm")
    print(
        f"Q1 ring radius: {ring_peak_radius*1e3:.4f} mm; "
        f"radial intensity FWHM: {ring_fwhm*1e3:.4f} mm"
    )
    print(f"Q1 centre/peak intensity: {ring_center_ratio:.6f}")
    print(
        f"Thickness ranges: point {np.nanmin(point_thickness)*1e3:.4f}–"
        f"{np.nanmax(point_thickness)*1e3:.4f} mm; ring "
        f"{np.nanmin(ring_thickness)*1e3:.4f}–{np.nanmax(ring_thickness)*1e3:.4f} mm"
    )


if __name__ == "__main__":
    main()
