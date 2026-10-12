from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
REPO_ROOT = PRINTHEAD_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))

from hologram3d_lib import (  # noqa: E402
    OrientedPlane,
    PlanarWallModel,
    evaluate_amplitude_loss,
    forward_pressure,
    initialize_from_focus_points,
    make_hologram_aperture,
    make_line_segment_target,
    make_optimization_samples,
    make_volume_grid,
    optimize_phase_adam,
)


RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "unobstructed_vertical_line_hologram.npz"
PLOT_PATH = RESULTS_DIR / "unobstructed_vertical_line_3d.png"


FREQUENCY_HZ = 2.28e6
SOUND_SPEED_M_S = 1500.0
OUTER_MEDIUM_DENSITY_KG_M3 = 1000.0
# Hydrogel assumption; update density when a measured formulation value is available.
INNER_MEDIUM_SOUND_SPEED_M_S = 1600.0
INNER_MEDIUM_DENSITY_KG_M3 = 1000.0
ACRYLIC_SOUND_SPEED_M_S = 2750.0
ACRYLIC_DENSITY_KG_M3 = 1175.0

HOLOGRAM_TO_ACRYLIC_OFFSET_M = 5e-3
ACRYLIC_THICKNESS_M = 2.68e-3


VOLUME_BOUNDS_M = ((0.0, 50e-3), (-25e-3, 25e-3), (-25e-3, 25e-3))
VOLUME_N = (100, 100, 100)  # (x, y, z) grid resolution

# Change the demanded shape here, then rerun this file.
DEMAND_SEGMENTS_M = [
    (np.array((20e-3, 0.0, -15e-3)), np.array((20e-3, -10e-3, 15e-3))),
    (np.array((20e-3, 0.0, -15e-3)), np.array((30e-3, 10e-3, 15e-3))) #,
   # (np.array((20e-3, 0.0, -15e-3)), np.array((30e-3, 0e-3, 15e-3)))
]
DEMAND_TUBE_RADIUS_M = 0.5e-3
PZT_COUNT = 2
PZT_N_SIDE = 100
PZT_DIAMETER_M = 50e-3
ACRYLIC_OUTER_FACE_POSITIONS_M = np.asarray([
    (0.0, 0.0, 0.0),
    (25e-3, 25e-3, 0.0),
])
INWARD_NORMALS = np.asarray([
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
])
TANGENT_HINTS = np.asarray([
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
])


def make_two_usual_actuators():
    if not (
        len(ACRYLIC_OUTER_FACE_POSITIONS_M)
        == len(INWARD_NORMALS)
        == len(TANGENT_HINTS)
        == PZT_COUNT
    ):
        raise ValueError("Define one wall position, normal, and tangent per PZT.")

    pzt_planes = [
        OrientedPlane.from_position_and_normal(
            f"PZT {chr(ord('A') + idx)}",
            position_m=(
                wall_position_m
                - HOLOGRAM_TO_ACRYLIC_OFFSET_M * inward_normal
            ),
            normal=inward_normal,
            tangent_hint=tangent_hint,
        )
        for idx, (wall_position_m, inward_normal, tangent_hint) in enumerate(
            zip(
                ACRYLIC_OUTER_FACE_POSITIONS_M,
                INWARD_NORMALS,
                TANGENT_HINTS,
            )
        )
    ]
    return [
        make_hologram_aperture(plane, n_side=PZT_N_SIDE, width_m=PZT_DIAMETER_M)
        for plane in pzt_planes
    ]


def circular_aperture_mask(aperture):
    radius_mm = 0.5 * PZT_DIAMETER_M * 1e3
    return aperture.uu_mm**2 + aperture.vv_mm**2 <= radius_mm**2


def make_wall_models(apertures):
    return [
        PlanarWallModel(
            frequency_hz=FREQUENCY_HZ,
            outer_sound_speed_m_s=SOUND_SPEED_M_S,
            outer_density_kg_m3=OUTER_MEDIUM_DENSITY_KG_M3,
            wall_offset_m=HOLOGRAM_TO_ACRYLIC_OFFSET_M,
            wall_thickness_m=ACRYLIC_THICKNESS_M,
            wall_sound_speed_m_s=ACRYLIC_SOUND_SPEED_M_S,
            wall_density_kg_m3=ACRYLIC_DENSITY_KG_M3,
            inner_sound_speed_m_s=INNER_MEDIUM_SOUND_SPEED_M_S,
            inner_density_kg_m3=INNER_MEDIUM_DENSITY_KG_M3,
        )
        for _ in apertures
    ]


def acrylic_face_positions(apertures):
    outer_faces_m = np.asarray([
        aperture.plane.position_m
        + HOLOGRAM_TO_ACRYLIC_OFFSET_M * aperture.plane.normal
        for aperture in apertures
    ])
    inner_faces_m = np.asarray([
        outer_face_m + ACRYLIC_THICKNESS_M * aperture.plane.normal
        for outer_face_m, aperture in zip(outer_faces_m, apertures)
    ])
    return outer_faces_m, inner_faces_m


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    aperture_masks = [circular_aperture_mask(aperture) for aperture in apertures]
    propagation_models = make_wall_models(apertures)
    acrylic_outer_faces_m, acrylic_inner_faces_m = acrylic_face_positions(apertures)
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
        propagation_models=propagation_models,
    )
    aperture_fields = [
        field * mask for field, mask in zip(aperture_fields, aperture_masks)
    ]
    _, initial_amp, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        aperture_fields,
        desired_amp,
        loss_weights,
        pressure_scale=1.0,
        wavelength_m=wavelength_m,
        target_extent_m=target_extent_m,
        propagation_models=propagation_models,
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
        propagation_models=propagation_models,
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
        n_iter=500,
        learning_rate=0.08,
        initial_aperture_fields=aperture_fields,
        aperture_masks=aperture_masks,
        propagation_models=propagation_models,
    )

    free_pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
        propagation_models=propagation_models,
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
        pzt_count=PZT_COUNT,
        pzt_n_side=PZT_N_SIDE,
        pzt_diameter_m=PZT_DIAMETER_M,
        pzt_width_m=PZT_DIAMETER_M,
        pzt_positions_m=np.asarray([
            aperture.plane.position_m for aperture in apertures
        ]),
        pzt_normals=np.asarray([
            aperture.plane.normal for aperture in apertures
        ]),
        pzt_tangent_hints=np.asarray([
            aperture.plane.tangent_u for aperture in apertures
        ]),
        aperture_masks=np.stack(aperture_masks, axis=0),
        acrylic_outer_face_positions_m=acrylic_outer_faces_m,
        acrylic_inner_face_positions_m=acrylic_inner_faces_m,
        propagation_model="parallel_fluid_acrylic_fluid_wall",
        hologram_to_acrylic_offset_m=HOLOGRAM_TO_ACRYLIC_OFFSET_M,
        acrylic_thickness_m=ACRYLIC_THICKNESS_M,
        acrylic_sound_speed_m_s=ACRYLIC_SOUND_SPEED_M_S,
        acrylic_density_kg_m3=ACRYLIC_DENSITY_KG_M3,
        acrylic_impedance_rayl=(
            ACRYLIC_DENSITY_KG_M3 * ACRYLIC_SOUND_SPEED_M_S
        ),
        outer_medium="water",
        outer_medium_sound_speed_m_s=SOUND_SPEED_M_S,
        outer_medium_density_kg_m3=OUTER_MEDIUM_DENSITY_KG_M3,
        outer_medium_impedance_rayl=(
            OUTER_MEDIUM_DENSITY_KG_M3 * SOUND_SPEED_M_S
        ),
        inner_medium="hydrogel",
        inner_medium_sound_speed_m_s=INNER_MEDIUM_SOUND_SPEED_M_S,
        inner_medium_density_kg_m3=INNER_MEDIUM_DENSITY_KG_M3,
        inner_medium_impedance_rayl=(
            INNER_MEDIUM_DENSITY_KG_M3 * INNER_MEDIUM_SOUND_SPEED_M_S
        ),
    )

    plot_3d_field(x, y, z, free_amp, target_amp, PLOT_PATH)
    print("Generated unobstructed hologram and field data")
    print("Target samples:", target_count)
    print("Initial loss:", initial_loss)
    print("Optimized loss:", history["loss"][-1])
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    print("Mean background |p| free:", float(free_amp[~target_mask].mean()))
    print("Hologram-to-acrylic offset [mm]:", HOLOGRAM_TO_ACRYLIC_OFFSET_M * 1e3)
    print("Acrylic thickness [mm]:", ACRYLIC_THICKNESS_M * 1e3)
    print(
        "Acrylic impedance [MRayl]:",
        ACRYLIC_DENSITY_KG_M3 * ACRYLIC_SOUND_SPEED_M_S / 1e6,
    )
    for aperture, outer_face_m, inner_face_m in zip(
        apertures,
        acrylic_outer_faces_m,
        acrylic_inner_faces_m,
    ):
        print(f"{aperture.plane.name} hologram [mm]:", aperture.plane.position_m * 1e3)
        print(f"{aperture.plane.name} acrylic outer face [mm]:", outer_face_m * 1e3)
        print(f"{aperture.plane.name} acrylic inner face [mm]:", inner_face_m * 1e3)
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
