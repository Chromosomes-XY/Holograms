from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))

from hologram3d_lib import (  # noqa: E402
    OrientedPlane,
    evaluate_amplitude_loss,
    forward_pressure,
    initialize_from_focus_points,
    make_hologram_aperture,
    make_line_segment_target,
    make_optimization_samples,
    make_volume_grid,
    optimize_phase_adam,
)


DATA_PATH = THIS_DIR / "unobstructed_vertical_line_hologram.npz"
PLOT_PATH = THIS_DIR / "unobstructed_vertical_line_3d.png"


FREQUENCY_HZ = 2.05e6
SOUND_SPEED_M_S = 1500.0
VOLUME_BOUNDS_M = ((0.0, 40e-3), (-20e-3, 20e-3), (-20e-3, 20e-3))
VOLUME_N = (49, 49, 49)

# Change the demanded shape here, then rerun this file.
DEMAND_SEGMENTS_M = [
    (np.array((20e-3, 0.0, -15e-3)), np.array((20e-3, 0.0, 15e-3))),
]
DEMAND_TUBE_RADIUS_M = 0.5e-3


def make_two_usual_actuators():
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
    return [
        make_hologram_aperture(plane, n_side=64, width_m=40e-3)
        for plane in pzt_planes
    ]


def main():
    wavelength_m = SOUND_SPEED_M_S / FREQUENCY_HZ
    x, y, z, points_m = make_volume_grid(VOLUME_BOUNDS_M, n=VOLUME_N)
    target_extent_m = max(x[-1] - x[0], y[-1] - y[0], z[-1] - z[0])
    target_amp = make_line_segment_target(
        points_m,
        DEMAND_SEGMENTS_M,
        tube_radius_m=DEMAND_TUBE_RADIUS_M,
    )
    target_mask = target_amp > 0.0
    if not np.any(target_mask):
        raise ValueError("Demand sampled zero target points. Increase tube radius or grid resolution.")

    apertures = make_two_usual_actuators()
    _, sample_points_m, desired_amp, loss_weights, target_count = make_optimization_samples(
        points_m,
        target_amp,
        max_background=4500,
    )

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
        n_iter=120,
        learning_rate=0.08,
        initial_aperture_fields=aperture_fields,
    )

    free_pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
    )
    free_amp = np.abs(free_pressure)

    aperture_fields_array = np.stack(aperture_fields, axis=0)
    demand_segments_array = np.array(DEMAND_SEGMENTS_M, dtype=float)
    np.savez_compressed(
        DATA_PATH,
        x=x,
        y=y,
        z=z,
        target_amp=target_amp,
        free_pressure=free_pressure,
        aperture_fields=aperture_fields_array,
        demand_segments_m=demand_segments_array,
        demand_tube_radius_m=DEMAND_TUBE_RADIUS_M,
        frequency_hz=FREQUENCY_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        target_extent_m=target_extent_m,
        pressure_scale=pressure_scale,
        initial_loss=initial_loss,
        optimized_loss=history["loss"][-1],
    )

    plot_3d_field(x, y, z, free_amp, target_amp, PLOT_PATH)
    print("Generated unobstructed hologram and field data")
    print("Target samples:", target_count)
    print("Initial loss:", initial_loss)
    print("Optimized loss:", history["loss"][-1])
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    print("Mean background |p| free:", float(free_amp[~target_mask].mean()))
    print(f"Saved data to {DATA_PATH}")
    print(f"Saved 3D plot to {PLOT_PATH}")


def plot_3d_field(x, y, z, pressure_amp, target_amp, output_path):
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_m = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    threshold = np.percentile(pressure_amp, 99.55)
    hot = pressure_amp >= threshold
    target = target_amp > 0.0

    fig = plt.figure(figsize=(7.0, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points_m[hot, 0] * 1e3,
        points_m[hot, 1] * 1e3,
        points_m[hot, 2] * 1e3,
        c=pressure_amp[hot],
        s=10,
        cmap="Reds",
        alpha=0.82,
        depthshade=False,
    )
    ax.scatter(
        points_m[target, 0] * 1e3,
        points_m[target, 1] * 1e3,
        points_m[target, 2] * 1e3,
        c="cyan",
        marker=".",
        s=10,
        alpha=0.45,
        depthshade=False,
    )
    ax.set_title("Unobstructed optimized field")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax.set_ylim(y[0] * 1e3, y[-1] * 1e3)
    ax.set_zlim(z[0] * 1e3, z[-1] * 1e3)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=22, azim=-55)
    fig.colorbar(scatter, ax=ax, shrink=0.72, label="|p|")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
