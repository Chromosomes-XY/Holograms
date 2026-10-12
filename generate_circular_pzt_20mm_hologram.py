"""Generate and evaluate a 40 mm circular-PZT point focus at 20 mm."""

from pathlib import Path

import numpy as np

from compare_tweezer_focal_distances import profile_width
from generate_square_pzt_hologram import (
    C_HOLOGRAM,
    C_MEDIUM,
    FREQUENCY_HZ,
    MAX_THICKNESS_M,
    MIN_THICKNESS_M,
    N,
    N_OPTIMIZATION_ITERATIONS,
    SIMULATION_WIDTH_M,
    TARGET_DOT_DIAMETER_M,
)
from hologramGen import (
    acoustic_params,
    apply_hologram,
    design_hologram_single_plane,
    impedance_params,
    make_center_dot_target,
    make_circular_pzt,
    make_grid,
    project_hologram,
    propagate_ASM,
)


PZT_DIAMETER_M = 40e-3
FOCAL_DISTANCE_M = 20e-3
N_AXIAL_SAMPLES = 181
OUTPUT_PATH = Path(
    "tweezer_focal_distance_comparison/circular_pzt_f20mm.npz"
)


def main():
    x, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    pzt_amp = make_circular_pzt(xv, yv, PZT_DIAMETER_M)
    target = make_center_dot_target(xv, yv, TARGET_DOT_DIAMETER_M)
    acoustic = acoustic_params(FREQUENCY_HZ, C_MEDIUM, C_HOLOGRAM)
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)

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
        Tmax=MAX_THICKNESS_M,
    )
    focal_field = project_hologram(
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
    focal_amp = np.abs(focal_field)
    centre = N // 2
    x_profile = focal_amp[centre]
    y_profile = focal_amp[:, centre]
    x_amp_fwhm_m = profile_width(x, x_profile, 0.5)[0]
    y_amp_fwhm_m = profile_width(x, y_profile, 0.5)[0]
    x_int_fwhm_m = profile_width(x, x_profile, np.sqrt(0.5))[0]
    y_int_fwhm_m = profile_width(x, y_profile, np.sqrt(0.5))[0]

    exit_field = apply_hologram(
        np.asarray(pzt_amp, dtype=np.complex128),
        thickness,
        impedance["Zt"],
        impedance["Zh"],
        impedance["Zm"],
        acoustic["kh"],
        acoustic["km"],
    )
    z_m = np.linspace(0.75 * FOCAL_DISTANCE_M, 1.25 * FOCAL_DISTANCE_M, N_AXIAL_SAMPLES)
    on_axis_amp = np.empty(z_m.size)
    xz_amp = np.empty((z_m.size, N))
    for index, distance_m in enumerate(z_m):
        plane_amp = np.abs(
            propagate_ASM(exit_field, dx, acoustic["lam_m"], float(distance_m))
        )
        xz_amp[index] = plane_amp[centre]
        on_axis_amp[index] = plane_amp[centre, centre]
    axial_peak_z_m = float(z_m[np.argmax(on_axis_amp)])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        x_m=x,
        z_m=z_m,
        pzt_mask=pzt_amp,
        thickness_m=thickness,
        focal_pressure_amplitude=focal_amp,
        xz_pressure_amplitude=xz_amp,
        on_axis_pressure_amplitude=on_axis_amp,
        focal_distance_m=FOCAL_DISTANCE_M,
        axial_peak_z_m=axial_peak_z_m,
        x_fwhm_amplitude_m=x_amp_fwhm_m,
        y_fwhm_amplitude_m=y_amp_fwhm_m,
        x_fwhm_intensity_m=x_int_fwhm_m,
        y_fwhm_intensity_m=y_int_fwhm_m,
        final_target_correlation=history["amp_corr"][-1],
    )
    print(f"Saved {OUTPUT_PATH}")
    print(f"Axial peak [mm]: {axial_peak_z_m * 1e3:.6f}")
    print(
        "Amplitude FWHM [mm]: "
        f"{x_amp_fwhm_m * 1e3:.6f} x {y_amp_fwhm_m * 1e3:.6f}"
    )
    print(
        "Intensity FWHM [mm]: "
        f"{x_int_fwhm_m * 1e3:.6f} x {y_int_fwhm_m * 1e3:.6f}"
    )


if __name__ == "__main__":
    main()
