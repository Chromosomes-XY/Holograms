from pathlib import Path

import numpy as np

from hologram3d_lib import (
    OrientedPlane,
    aperture_visibility_report,
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
    radius_m=10e-3,
    turns=3,
    n_points=140,
):
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_points)
    x = np.linspace(x_start_m, x_end_m, n_points)
    y = radius_m * np.cos(theta)
    z = radius_m * np.sin(theta)
    curve = np.column_stack((x, y, z))
    return [(curve[idx], curve[idx + 1]) for idx in range(curve.shape[0] - 1)], curve


def make_three_hologram_planes(center_m, triangle_radius_m=20e-3):
    vertex_angles = np.deg2rad((180.0, 300.0, 60.0))
    center_m = np.asarray(center_m, dtype=float)
    centers_m = [
        center_m
        + triangle_radius_m * np.array((np.cos(angle), np.sin(angle), 0.0))
        for angle in vertex_angles
    ]
    normals = [
        (center_m - pzt_center_m) / np.linalg.norm(center_m - pzt_center_m)
        for pzt_center_m in centers_m
    ]
    return [
        OrientedPlane.from_position_and_normal(
            f"PZT {chr(ord('A') + idx)}",
            position_m=pzt_center_m,
            normal=normal,
        )
        for idx, (pzt_center_m, normal) in enumerate(zip(centers_m, normals))
    ]


def assert_three_hologram_geometry(apertures, expected_centroid_m):
    if len(apertures) != 3:
        raise ValueError("Expected exactly three holograms.")
    if not all(np.isclose(aperture.width_m, 40e-3) for aperture in apertures):
        raise ValueError("Each hologram must be a 40 mm square.")
    centers_m = np.array([aperture.plane.position_m for aperture in apertures])
    if not np.allclose(centers_m.mean(axis=0), expected_centroid_m):
        raise ValueError("Hologram centres must form a triangle around the target centre.")
    side_lengths_m = np.array(
        [
            np.linalg.norm(centers_m[0] - centers_m[1]),
            np.linalg.norm(centers_m[1] - centers_m[2]),
            np.linalg.norm(centers_m[2] - centers_m[0]),
        ]
    )
    if not np.allclose(side_lengths_m, side_lengths_m[0]):
        raise ValueError("Hologram centres must be equilateral triangle vertices.")
    for idx, left in enumerate(apertures):
        inward = expected_centroid_m - left.plane.position_m
        inward = inward / np.linalg.norm(inward)
        if not np.allclose(left.plane.normal, inward):
            raise ValueError("Each hologram normal must point toward the triangle centre.")
        for right in apertures[idx + 1 :]:
            angle_deg = np.degrees(
                np.arccos(
                    np.clip(np.dot(left.plane.normal, right.plane.normal), -1.0, 1.0)
                )
            )
            if not np.isclose(angle_deg, 120.0):
                raise ValueError("Hologram normals must be 120 degrees apart.")


def main():
    frequency_hz = 2.05e6
    sound_speed_m_s = 1500.0
    wavelength_m = sound_speed_m_s / frequency_hz

    # Three adjacent 40 mm square holograms around the spiral volume.
    center_m = np.array((20e-3, 0.0, 0.0))
    pzt_planes = make_three_hologram_planes(center_m)
    apertures = [
        make_hologram_aperture(plane, n_side=64, width_m=40e-3)
        for plane in pzt_planes
    ]
    assert_three_hologram_geometry(apertures, center_m)

    # Demand: a helix through [1, 3] x [-1, 1] x [-1, 1] cm.
    spiral_segments_m, spiral_points_m = make_spiral_segments()
    spiral_tube_radius_m = 0.7e-3
    target_center_m = spiral_points_m.mean(axis=0)
    distance_bin_m = 0.75e-3

    x, y, z, points_m = make_volume_grid(
        bounds_m=((0.0, 40e-3), (-20e-3, 20e-3), (-20e-3, 20e-3)),
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
        distance_bin_m=distance_bin_m,
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
        distance_bin_m=distance_bin_m,
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
        distance_bin_m=distance_bin_m,
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
        n_iter=100,
        learning_rate=0.15,
        initial_aperture_fields=aperture_fields,
        distance_bin_m=distance_bin_m,
    )

    pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=distance_bin_m,
    )
    pressure_amp = np.abs(pressure)

    output_path = SCRIPT_DIR / "three_holograms_spiral_pressure.png"
    plot_pressure_percentile = 99.0
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
    pzt_centers_m = np.array([aperture.plane.position_m for aperture in apertures])
    pzt_side_lengths_m = np.array(
        [
            np.linalg.norm(pzt_centers_m[0] - pzt_centers_m[1]),
            np.linalg.norm(pzt_centers_m[1] - pzt_centers_m[2]),
            np.linalg.norm(pzt_centers_m[2] - pzt_centers_m[0]),
        ]
    )
    print("PZT triangle centroid [m]:", pzt_centers_m.mean(axis=0))
    print("PZT triangle side lengths [m]:", pzt_side_lengths_m)
    print("Plot/evaluation domain [m]: x=[0.0, 0.04], y=[-0.02, 0.02], z=[-0.02, 0.02]")
    print("Demand spiral domain [m]: x=[0.01, 0.03], y=[-0.01, 0.01], z=[-0.01, 0.01]")
    print("Propagation distance bin [m]:", distance_bin_m)
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
