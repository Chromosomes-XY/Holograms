from pathlib import Path

import numpy as np

from hologram3d_lib import (
    OrientedPlane,
    aperture_visibility_report,
    assert_geometry,
    evaluate_amplitude_loss,
    forward_pressure,
    initialize_from_focus_points,
    make_hologram_aperture,
    make_line_segment_target,
    make_optimization_samples,
    make_volume_grid,
    optimize_phase_adam,
    plot_results,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def make_spiral_segments(
    x_start_m=10e-3,
    x_end_m=30e-3,
    radius_m=6e-3,
    turns=1,
    n_points=140,
):
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_points)
    x = np.linspace(x_start_m, x_end_m, n_points)
    y = radius_m * np.cos(theta)
    z = radius_m * np.sin(theta)
    curve = np.column_stack((x, y, z))
    return [(curve[idx], curve[idx + 1]) for idx in range(curve.shape[0] - 1)], curve


def main():
    frequency_hz = 2.05e6
    sound_speed_m_s = 1500.0
    wavelength_m = sound_speed_m_s / frequency_hz

    pzt_planes = [
        OrientedPlane.from_position_and_normal(
            "PZT A",
            position_m=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
        ),
        OrientedPlane.from_position_and_normal(
            "PZT B",
            position_m=(20e-3, 20e-3, 0.0),
            normal=(0.0, -1.0, 0.0),
            tangent_hint=(1.0, 0.0, 0.0),
        ),
    ]
    apertures = [
        make_hologram_aperture(plane, n_side=64, width_m=40e-3)
        for plane in pzt_planes
    ]
    assert_geometry(apertures)

    # Demand: a helix through [1, 3] x [-1, 1] x [-1, 1] cm.
    spiral_segments_m, spiral_points_m = make_spiral_segments()
    spiral_tube_radius_m = 0.7e-3
    target_center_m = spiral_points_m.mean(axis=0)

    x, y, z, points_m = make_volume_grid(
        bounds_m=((10e-3, 30e-3), (-10e-3, 10e-3), (-10e-3, 10e-3)),
        n=(52, 52, 52),
    )
    target_amp = make_line_segment_target(
        points_m,
        spiral_segments_m,
        spiral_tube_radius_m,
    )
    _, sample_points_m, desired_amp, loss_weights, target_count = make_optimization_samples(
        points_m,
        target_amp,
        max_background=6000,
    )
    if target_count == 0:
        raise ValueError(
            "The spiral produced zero target samples. Increase "
            "spiral_tube_radius_m or the grid resolution."
        )

    target_extent_m = max(x[-1] - x[0], y[-1] - y[0], z[-1] - z[0])
    aperture_fields = initialize_from_focus_points(
        apertures,
        sample_points_m[:target_count],
        wavelength_m,
        target_extent_m=target_extent_m,
    )

    _, initial_amp, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        aperture_fields,
        desired_amp,
        loss_weights,
        pressure_scale=1.0,
        wavelength_m=wavelength_m,
        target_extent_m=target_extent_m,
    )
    pressure_scale = float(initial_amp[:target_count].mean())
    initial_loss, _, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        aperture_fields,
        desired_amp,
        loss_weights,
        pressure_scale=pressure_scale,
        wavelength_m=wavelength_m,
        target_extent_m=target_extent_m,
    )
    aperture_fields, history = optimize_phase_adam(
        sample_points_m,
        apertures,
        desired_amp,
        loss_weights,
        pressure_scale,
        target_count,
        wavelength_m,
        target_extent_m,
        n_iter=200,
        learning_rate=0.06,
        initial_aperture_fields=aperture_fields,
    )

    pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
    )
    pressure_amp = np.abs(pressure)

    output_path = SCRIPT_DIR / "two_holograms_spiral_pressure.png"
    plot_pressure_percentile = 99.
    plot_results(
        x=x,
        y=y,
        z=z,
        pressure_amp=pressure_amp,
        target_amp=target_amp,
        apertures=apertures,
        aperture_fields=aperture_fields,
        target_center_m=target_center_m,
        output_path=output_path,
        plot_pressure_percentile=plot_pressure_percentile,
    )

    target_mask = target_amp > 0.0
    print("Boundary:")
    for aperture in apertures:
        print(
            f"  {aperture.plane.name}: position={aperture.plane.position_m}, "
            f"normal={aperture.plane.normal}, side={aperture.width_m}"
        )
    print("Demand spiral domain [m]: x=[0.01, 0.03], y=[-0.01, 0.01], z=[-0.01, 0.01]")
    print("Spiral segments:", len(spiral_segments_m))
    print("Target sample count:", target_count)
    for report in aperture_visibility_report(apertures, points_m[target_mask]):
        print(
            f"  {report['name']} visibility: "
            f"front={report['front_count']}/{report['total_count']}, "
            f"inside_aperture={report['aperture_count']}/{report['total_count']}, "
            f"usable={report['usable_count']}/{report['total_count']}"
        )
    print("Initial focused-field loss:", initial_loss)
    print("Optimized loss:", history["loss"][-1])
    print("Mean pressure on spiral:", float(pressure_amp[target_mask].mean()))
    print("Mean pressure off spiral:", float(pressure_amp[~target_mask].mean()))
    print(
        "Plot threshold:",
        f"top {100.0 - plot_pressure_percentile:.2f}% "
        f"(|p| >= {np.percentile(pressure_amp, plot_pressure_percentile):.6g})",
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
