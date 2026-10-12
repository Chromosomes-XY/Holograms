from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
REPO_ROOT = PRINTHEAD_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))
sys.path.insert(0, str(PRINTHEAD_DIR / "lib"))

from hologram3d_lib import (  # noqa: E402
    OrientedPlane,
    adjoint_propagate,
    bilinear_adjoint,
    bilinear_sample,
    distance_groups,
    forward_pressure,
    initialize_from_focus_points,
    make_hologram_aperture,
    make_line_segment_target,
    make_volume_grid,
    point_local_coordinates,
    propagate_aperture,
)
from occlusion_lib import ShellAndBottomPrintHead, sampled_aperture_visibility  # noqa: E402


RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "occlusion_optimized_vertical_line_hologram.npz"


# Header: change these to define the volume, boundary, demanded field, and head sweep.
FREQUENCY_HZ = 2.05e6
SOUND_SPEED_M_S = 1500.0
VOLUME_BOUNDS_M = ((0.0, 50e-3), (-25e-3, 25e-3), (-25e-3, 25e-3))
VOLUME_N = (64, 64, 64)
TOP_DEMAND_Z_MIN_M = -5e-3
TOP_DEMAND_Z_MAX_M = 20e-3
DEMAND_SEGMENTS_M = [
    (np.array((25e-3, 0.0, TOP_DEMAND_Z_MIN_M)), np.array((25e-3, -10e-3, TOP_DEMAND_Z_MAX_M))),
     (np.array((25e-3, 0.0, TOP_DEMAND_Z_MIN_M)), np.array((25e-3, 10e-3, TOP_DEMAND_Z_MAX_M))),
       (np.array((25e-3, 0.0, TOP_DEMAND_Z_MIN_M)), np.array((15e-3, 0.0, TOP_DEMAND_Z_MAX_M))),
]
DEMAND_TUBE_RADIUS_M = 0.75e-3

OCCLUSION_LEVEL_COUNT = 10
OCCLUSION_FRACTIONS = tuple(np.linspace(0.66, 0.0, OCCLUSION_LEVEL_COUNT))
LEVEL_SLAB_THICKNESS_M = 0.6e-3
HEAD_RADIUS_M = 15e-3
HEAD_LENGTH_M = 45e-3
SHELL_SOFT_EDGE_M = 1.0e-3
SHELL_TRANSMISSION = 0.0
BOTTOM_SOFT_EDGE_M = 0.5e-3
BOTTOM_TRANSMISSION = 0.04
BOTTOM_PRESSURE_REFLECTION = -0.98 + 0.0j
VISIBILITY_SOURCE_STRIDE = 4

PZT_N_SIDE = 48
PZT_DIAMETER_M = 50e-3
PZT_WIDTH_M = PZT_DIAMETER_M
MAX_BACKGROUND_SAMPLES_PER_LEVEL = 80
N_ITER = 120
LEARNING_RATE = 0.01
DISTANCE_BIN_M = 1.0e-3


def make_three_actuators():
    pzt_planes = [
        OrientedPlane.from_position_and_normal(
            "PZT A",
            position_m=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
        ),
        OrientedPlane.from_position_and_normal(
            "PZT B",
            position_m=(27e-3, 27e-3, 0.0),
            normal=(0.0, -1.0, 0.0),
            tangent_hint=(1.0, 0.0, 0.0),
        ),
        OrientedPlane.from_position_and_normal(
            "PZT C",
            position_m=(0.0, 27e-3, -27e-3),
            normal=(0.0, 0.0, 1.0),
            tangent_hint=(1.0, 0.0, 0.0),
        ),
    ]
    return [
        make_hologram_aperture(plane, n_side=PZT_N_SIDE, width_m=PZT_WIDTH_M)
        for plane in pzt_planes
    ]


def circular_aperture_mask(aperture):
    return (aperture.uu_mm**2 + aperture.vv_mm**2) <= (0.5 * PZT_DIAMETER_M * 1e3) ** 2


def apply_circular_apertures(apertures, aperture_fields):
    return [
        field * circular_aperture_mask(aperture)
        for aperture, field in zip(apertures, aperture_fields)
    ]


def make_print_head_for_occlusion_fraction(fraction, volume_z_min_m, volume_z_max_m, head_xy_m):
    volume_depth_m = volume_z_max_m - volume_z_min_m
    tip_z_m = volume_z_max_m - fraction * volume_depth_m
    return ShellAndBottomPrintHead.from_tip_position(
        tip_m=(head_xy_m[0], head_xy_m[1], tip_z_m),
        axis=(0.0, 0.0, 1.0),
        radius_m=HEAD_RADIUS_M,
        length_m=HEAD_LENGTH_M,
        shell_soft_edge_m=SHELL_SOFT_EDGE_M,
        shell_transmission=SHELL_TRANSMISSION,
        bottom_soft_edge_m=BOTTOM_SOFT_EDGE_M,
        bottom_transmission=BOTTOM_TRANSMISSION,
        bottom_pressure_reflection=BOTTOM_PRESSURE_REFLECTION,
    )


def make_visibility_stack(apertures, points_m, print_heads):
    visibility_stack = []
    diagnostics_stack = []
    for print_head in print_heads:
        visibility = np.zeros((len(apertures), points_m.shape[0]), dtype=float)
        diagnostics = []
        for idx, aperture in enumerate(apertures):
            visibility[idx], aperture_diagnostics = sampled_aperture_visibility(
                aperture,
                points_m,
                print_head,
                stride=VISIBILITY_SOURCE_STRIDE,
            )
            diagnostics.append(aperture_diagnostics)
        visibility_stack.append(visibility)
        diagnostics_stack.append(diagnostics)
    return np.asarray(visibility_stack), diagnostics_stack


def make_slab_optimization_samples(
    points_m,
    target_amp,
    slab_z_min_m,
    slab_z_max_m,
    max_background,
    seed,
):
    rng = np.random.default_rng(seed)
    in_slab = (points_m[:, 2] >= slab_z_min_m) & (points_m[:, 2] <= slab_z_max_m)
    target_idx = np.flatnonzero(in_slab & (target_amp > 0.0))
    background_idx = np.flatnonzero(in_slab & (target_amp <= 0.0))
    if background_idx.size > max_background:
        background_idx = rng.choice(background_idx, size=max_background, replace=False)

    sample_idx = np.concatenate((target_idx, background_idx))
    desired_amp = np.zeros(sample_idx.size, dtype=float)
    loss_weights = np.full(sample_idx.size, 0.05, dtype=float)
    desired_amp[: target_idx.size] = 1.0
    loss_weights[: target_idx.size] = 80.0
    return {
        "sample_idx": sample_idx,
        "points_m": points_m[sample_idx],
        "desired_amp": desired_amp,
        "loss_weights": loss_weights,
        "target_count": int(target_idx.size),
        "background_count": int(background_idx.size),
        "slab_z_min_m": slab_z_min_m,
        "slab_z_max_m": slab_z_max_m,
    }


def forward_pressure_with_visibility(
    points_m,
    apertures,
    aperture_fields,
    visibility,
    wavelength_m,
    target_extent_m,
    distance_bin_m=DISTANCE_BIN_M,
):
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    for aperture_idx, (aperture, aperture_field) in enumerate(zip(apertures, aperture_fields)):
        active_aperture_field = aperture_field * circular_aperture_mask(aperture)
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        forward = s > 0
        for distance, slice_mask in distance_groups(s, forward, distance_bin_m):
            propagated = propagate_aperture(
                active_aperture_field,
                aperture,
                wavelength_m,
                float(distance),
                target_extent_m,
            )
            values, *_ = bilinear_sample(
                propagated,
                aperture.coords_m,
                u[slice_mask],
                v[slice_mask],
            )
            pressure[slice_mask] += visibility[aperture_idx, slice_mask] * values
    return pressure


def occlusion_loss_and_gradient(
    level_records,
    apertures,
    aperture_fields,
    pressure_scale,
    wavelength_m,
    target_extent_m,
):
    eps = 1e-12
    normalizer = sum(np.sum(record["loss_weights"]) for record in level_records)
    total_loss = 0.0
    target_amp_sum = 0.0
    target_count_sum = 0
    background_amp_sum = 0.0
    background_count_sum = 0
    aperture_grads = [np.zeros(aperture.shape, dtype=np.complex128) for aperture in apertures]

    for record in level_records:
        points_m = record["points_m"]
        desired_amp = record["desired_amp"]
        loss_weights = record["loss_weights"]
        visibility = record["visibility"]
        target_count = record["target_count"]
        pressure = forward_pressure_with_visibility(
            points_m,
            apertures,
            aperture_fields,
            visibility,
            wavelength_m,
            target_extent_m,
        )
        amp = np.abs(pressure)
        residual = amp / pressure_scale - desired_amp
        total_loss += float(np.sum(loss_weights * residual**2) / normalizer)
        if target_count > 0:
            target_amp_sum += float(np.sum(amp[:target_count]))
            target_count_sum += target_count
        if amp.size > target_count:
            background_amp_sum += float(np.sum(amp[target_count:]))
            background_count_sum += amp.size - target_count
        grad_pressure = (
            (2.0 * loss_weights / normalizer)
            * residual
            * pressure
            / ((amp + eps) * pressure_scale)
        )

        for aperture_idx, aperture in enumerate(apertures):
            grad_aperture = np.zeros(aperture.shape, dtype=np.complex128)
            u, v, s = point_local_coordinates(points_m, aperture.plane)
            forward = s > 0
            for distance, slice_mask in distance_groups(s, forward, DISTANCE_BIN_M):
                _, valid, i0, j0, du, dv = bilinear_sample(
                    np.zeros(aperture.shape, dtype=np.complex128),
                    aperture.coords_m,
                    u[slice_mask],
                    v[slice_mask],
                )
                slice_grad = bilinear_adjoint(
                    visibility[aperture_idx, slice_mask] * grad_pressure[slice_mask],
                    valid,
                    i0,
                    j0,
                    du,
                    dv,
                    aperture.shape,
                )
                grad_aperture += adjoint_propagate(
                    slice_grad,
                    aperture,
                    wavelength_m,
                    float(distance),
                    target_extent_m,
                )
            aperture_grads[aperture_idx] += grad_aperture

    target_mean = target_amp_sum / max(target_count_sum, 1)
    background_mean = background_amp_sum / max(background_count_sum, 1)
    return total_loss, target_mean, background_mean, aperture_grads


def optimize_for_occlusion_levels(
    level_records,
    apertures,
    pressure_scale,
    wavelength_m,
    target_extent_m,
    initial_aperture_fields,
):
    phases = [np.angle(field) for field in initial_aperture_fields]
    m = [np.zeros(aperture.shape, dtype=float) for aperture in apertures]
    v = [np.zeros(aperture.shape, dtype=float) for aperture in apertures]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history = {"loss": [], "target_mean": [], "background_mean": []}

    for iteration in range(1, N_ITER + 1):
        aperture_fields = apply_circular_apertures(
            apertures,
            [np.exp(1j * phase) for phase in phases],
        )
        loss, target_mean, background_mean, aperture_grads = occlusion_loss_and_gradient(
            level_records,
            apertures,
            aperture_fields,
            pressure_scale,
            wavelength_m,
            target_extent_m,
        )
        for idx, (phase, field, grad_field) in enumerate(zip(phases, aperture_fields, aperture_grads)):
            grad_field = grad_field * circular_aperture_mask(apertures[idx])
            grad_phase = np.real(np.conj(field) * (1j * grad_field))
            m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad_phase
            v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad_phase**2
            m_hat = m[idx] / (1.0 - beta1**iteration)
            v_hat = v[idx] / (1.0 - beta2**iteration)
            phases[idx] = phase + LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)
            phases[idx] = (phases[idx] + np.pi) % (2.0 * np.pi) - np.pi

        history["loss"].append(loss)
        history["target_mean"].append(float(target_mean))
        history["background_mean"].append(float(background_mean))
        if iteration == 1 or iteration % 10 == 0:
            print(
                f"iter {iteration:03d} loss={loss:.6e} "
                f"target_mean={history['target_mean'][-1]:.6e} "
                f"background_mean={history['background_mean'][-1]:.6e}"
            )

    return [np.exp(1j * phase) for phase in phases], history


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

    apertures = make_three_actuators()
    head_xy_m = np.array(((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0))
    print_heads = [
        make_print_head_for_occlusion_fraction(fraction, z[0], z[-1], head_xy_m)
        for fraction in OCCLUSION_FRACTIONS
    ]

    level_records = []
    for level_idx, (fraction, print_head) in enumerate(zip(OCCLUSION_FRACTIONS, print_heads)):
        slab_z_max_m = float(print_head.tip_center_m[2])
        slab_z_min_m = slab_z_max_m - LEVEL_SLAB_THICKNESS_M
        record = make_slab_optimization_samples(
            points_m,
            target_amp,
            slab_z_min_m=slab_z_min_m,
            slab_z_max_m=slab_z_max_m,
            max_background=MAX_BACKGROUND_SAMPLES_PER_LEVEL,
            seed=10_000 + level_idx,
        )
        if record["points_m"].size == 0:
            continue
        visibility, _ = make_visibility_stack(apertures, record["points_m"], [print_head])
        record["visibility"] = visibility[0]
        record["fraction"] = fraction
        record["head_tip_m"] = print_head.tip_center_m
        level_records.append(record)

    if not level_records:
        raise ValueError("No optimization samples were generated in the print-head slabs.")

    all_target_points = [
        record["points_m"][: record["target_count"]]
        for record in level_records
        if record["target_count"] > 0
    ]
    if not all_target_points:
        raise ValueError("The 1 mm print-head slabs contain no demanded target samples.")
    initial_target_points_m = np.concatenate(all_target_points, axis=0)

    initial_aperture_fields = initialize_from_focus_points(
        apertures,
        initial_target_points_m,
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=DISTANCE_BIN_M,
    )
    initial_fields = apply_circular_apertures(
        apertures,
        [np.exp(1j * np.angle(field)) for field in initial_aperture_fields],
    )
    initial_target_means = []
    for record in level_records:
        if record["target_count"] == 0:
            continue
        pressure = forward_pressure_with_visibility(
            record["points_m"],
            apertures,
            initial_fields,
            record["visibility"],
            wavelength_m,
            target_extent_m,
        )
        initial_target_means.append(float(np.abs(pressure[: record["target_count"]]).mean()))
    pressure_scale = float(np.mean(initial_target_means))
    aperture_fields, history = optimize_for_occlusion_levels(
        level_records,
        apertures,
        pressure_scale,
        wavelength_m,
        target_extent_m,
        initial_fields,
    )
    aperture_fields = apply_circular_apertures(apertures, aperture_fields)

    free_pressure = forward_pressure(
        points_m,
        apertures,
        apply_circular_apertures(apertures, aperture_fields),
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=DISTANCE_BIN_M,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_PATH,
        x=x,
        y=y,
        z=z,
        target_amp=target_amp,
        free_pressure=free_pressure,
        aperture_fields=np.stack(aperture_fields, axis=0),
        occlusion_fractions=np.asarray(OCCLUSION_FRACTIONS),
        head_tip_positions_m=np.asarray([head.tip_center_m for head in print_heads]),
        level_slab_thickness_m=LEVEL_SLAB_THICKNESS_M,
        level_target_counts=np.asarray([record["target_count"] for record in level_records]),
        level_background_counts=np.asarray([record["background_count"] for record in level_records]),
        volume_bounds_m=np.asarray(VOLUME_BOUNDS_M, dtype=float),
        pzt_n_side=PZT_N_SIDE,
        pzt_width_m=PZT_WIDTH_M,
        pzt_diameter_m=PZT_DIAMETER_M,
        pzt_count=len(apertures),
        pzt_positions_m=np.asarray([aperture.plane.position_m for aperture in apertures]),
        pzt_normals=np.asarray([aperture.plane.normal for aperture in apertures]),
        pzt_tangent_hints=np.asarray([aperture.plane.tangent_u for aperture in apertures]),
        head_radius_m=HEAD_RADIUS_M,
        head_length_m=HEAD_LENGTH_M,
        shell_soft_edge_m=SHELL_SOFT_EDGE_M,
        shell_transmission=SHELL_TRANSMISSION,
        bottom_soft_edge_m=BOTTOM_SOFT_EDGE_M,
        bottom_transmission=BOTTOM_TRANSMISSION,
        bottom_pressure_reflection=BOTTOM_PRESSURE_REFLECTION,
        visibility_source_stride=VISIBILITY_SOURCE_STRIDE,
        distance_bin_m=DISTANCE_BIN_M,
        demand_segments_m=np.array(DEMAND_SEGMENTS_M, dtype=float),
        demand_tube_radius_m=DEMAND_TUBE_RADIUS_M,
        frequency_hz=FREQUENCY_HZ,
        sound_speed_m_s=SOUND_SPEED_M_S,
        target_extent_m=target_extent_m,
        pressure_scale=pressure_scale,
        optimized_loss=history["loss"][-1],
    )

    print("Optimized hologram for multiple print-head occlusion levels")
    print("Occlusion levels:", len(OCCLUSION_FRACTIONS))
    print("Active slab records:", len(level_records))
    print("Total target samples:", sum(record["target_count"] for record in level_records))
    print("Total background samples:", sum(record["background_count"] for record in level_records))
    print("Slab thickness [mm]:", LEVEL_SLAB_THICKNESS_M * 1e3)
    print("Occlusion fraction range:", (OCCLUSION_FRACTIONS[0], OCCLUSION_FRACTIONS[-1]))
    print("Optimized loss:", history["loss"][-1])
    print("Final aggregate target mean:", history["target_mean"][-1])
    print("Final aggregate background mean:", history["background_mean"][-1])
    print(f"Saved data to {DATA_PATH}")


if __name__ == "__main__":
    main()
