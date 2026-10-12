from pathlib import Path
import importlib.util
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
REPO_ROOT = PRINTHEAD_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))
sys.path.insert(0, str(PRINTHEAD_DIR / "lib"))

from hologram3d_lib import (  # noqa: E402
    OrientedPlane,
    PlanarWallModel,
    forward_pressure,
    make_hologram_aperture,
)
from occlusion_lib import (  # noqa: E402
    ShellAndBottomPrintHead,
    sampled_aperture_visibility,
)
from hologramGen import acoustic_params  # noqa: E402


RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "unobstructed_vertical_line_hologram.npz"
OUTPUT_PATH = RESULTS_DIR / "saved_hologram_occlusion_slices.png"
HEIGHT_PLOT_PATH = RESULTS_DIR / "hologram_height_arrays.png"

# Your current print-head parameters. Change these here for occlusion studies.
HEAD_RADIUS_M = 15e-3
HEAD_LENGTH_M = 45e-3
SHELL_SOFT_EDGE_M = 1.0e-3
SHELL_TRANSMISSION = 0.0
BOTTOM_SOFT_EDGE_M = 0.5e-3
BOTTOM_TRANSMISSION = 0.04
BOTTOM_PRESSURE_REFLECTION = -0.98 + 0.0j
SLICE_BELOW_INTERFACE_M = 0.5e-3
VISIBILITY_SAMPLES_PER_SIDE = 16
C_HOLOGRAM_M_S = 2550.0
BASE_THICKNESS_M = 0.4e-3


GEOMETRY_KEYS = {
    "pzt_normals",
    "pzt_positions_m",
    "pzt_tangent_hints",
    "pzt_width_m",
}


def load_baseline_generator():
    generator_path = THIS_DIR / "01_generate_unobstructed_vertical_line.py"
    spec = importlib.util.spec_from_file_location("free_field_baseline_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_actuators(data):
    pzt_count = int(data["pzt_count"])
    pzt_n_side = int(data["pzt_n_side"])
    data_keys = set(data.files if hasattr(data, "files") else data.keys())
    if GEOMETRY_KEYS.issubset(data_keys):
        pzt_width_m = float(data["pzt_width_m"])
        pzt_positions_m = data["pzt_positions_m"]
        pzt_normals = data["pzt_normals"]
        pzt_tangent_hints = data["pzt_tangent_hints"]
    else:
        baseline_generator = load_baseline_generator()
        template_apertures = baseline_generator.make_two_usual_actuators()
        if len(template_apertures) != pzt_count:
            raise ValueError(
                "Saved actuator count does not match the geometry defined in step 01."
            )
        saved_diameter_m = float(data["pzt_diameter_m"])
        if not np.isclose(saved_diameter_m, baseline_generator.PZT_DIAMETER_M):
            raise ValueError(
                "Saved actuator diameter does not match the geometry defined in step 01."
            )
        pzt_width_m = saved_diameter_m
        pzt_positions_m = np.asarray([
            aperture.plane.position_m for aperture in template_apertures
        ])
        pzt_normals = np.asarray([
            aperture.plane.normal for aperture in template_apertures
        ])
        pzt_tangent_hints = np.asarray([
            aperture.plane.tangent_u for aperture in template_apertures
        ])
    if not (
        len(pzt_positions_m)
        == len(pzt_normals)
        == len(pzt_tangent_hints)
        == pzt_count
    ):
        raise ValueError("Saved actuator geometry does not match pzt_count.")

    pzt_planes = [
        OrientedPlane.from_position_and_normal(
            f"PZT {chr(ord('A') + idx)}",
            position_m=position_m,
            normal=normal,
            tangent_hint=tangent_hint,
        )
        for idx, (position_m, normal, tangent_hint) in enumerate(
            zip(pzt_positions_m, pzt_normals, pzt_tangent_hints)
        )
    ]
    return [
        make_hologram_aperture(plane, n_side=pzt_n_side, width_m=pzt_width_m)
        for plane in pzt_planes
    ]


def circular_aperture_mask(aperture, diameter_m):
    radius_mm = 0.5 * diameter_m * 1e3
    return aperture.uu_mm**2 + aperture.vv_mm**2 <= radius_mm**2


def make_propagation_models(data, apertures):
    acrylic_impedance_rayl = (
        float(data["acrylic_density_kg_m3"])
        * float(data["acrylic_sound_speed_m_s"])
    )
    if not np.isclose(acrylic_impedance_rayl, float(data["acrylic_impedance_rayl"])):
        raise ValueError("Saved acrylic impedance is inconsistent with its density and speed.")
    return [
        PlanarWallModel(
            frequency_hz=float(data["frequency_hz"]),
            outer_sound_speed_m_s=float(data["outer_medium_sound_speed_m_s"]),
            outer_density_kg_m3=float(data["outer_medium_density_kg_m3"]),
            wall_offset_m=float(data["hologram_to_acrylic_offset_m"]),
            wall_thickness_m=float(data["acrylic_thickness_m"]),
            wall_sound_speed_m_s=float(data["acrylic_sound_speed_m_s"]),
            wall_density_kg_m3=float(data["acrylic_density_kg_m3"]),
            inner_sound_speed_m_s=float(data["inner_medium_sound_speed_m_s"]),
            inner_density_kg_m3=float(data["inner_medium_density_kg_m3"]),
        )
        for _ in apertures
    ]


def validate_acrylic_face_positions(data, apertures, propagation_models):
    saved_outer_faces_m = data["acrylic_outer_face_positions_m"]
    saved_inner_faces_m = data["acrylic_inner_face_positions_m"]
    calculated_outer_faces_m = np.asarray([
        aperture.plane.position_m
        + model.wall_offset_m * aperture.plane.normal
        for aperture, model in zip(apertures, propagation_models)
    ])
    calculated_inner_faces_m = np.asarray([
        outer_face_m + model.wall_thickness_m * aperture.plane.normal
        for outer_face_m, aperture, model in zip(
            calculated_outer_faces_m,
            apertures,
            propagation_models,
        )
    ])
    if not np.allclose(saved_outer_faces_m, calculated_outer_faces_m):
        raise ValueError("Saved acrylic outer faces do not match the hologram offsets.")
    if not np.allclose(saved_inner_faces_m, calculated_inner_faces_m):
        raise ValueError("Saved acrylic inner faces do not match the wall thickness.")
    return saved_outer_faces_m, saved_inner_faces_m


def phase_to_printable_height_m(phase_rad, frequency_hz, sound_speed_m_s):
    acoustic = acoustic_params(
        frequency_hz=frequency_hz,
        c_medium=sound_speed_m_s,
        c_hologram=C_HOLOGRAM_M_S,
    )
    delta_k = acoustic["km"] - acoustic["kh"]
    if delta_k <= 0.0:
        raise ValueError("Hologram material speed must exceed medium speed.")
    return BASE_THICKNESS_M + np.mod(phase_rad, 2.0 * np.pi) / delta_k


def plot_hologram_heights(
    apertures,
    aperture_fields,
    aperture_masks,
    frequency_hz,
    sound_speed_m_s,
    diameter_m,
    output_path,
):
    height_maps_mm = np.asarray([
        np.where(
            active_mask,
            phase_to_printable_height_m(
                np.angle(aperture_field),
                frequency_hz=frequency_hz,
                sound_speed_m_s=sound_speed_m_s,
            ) * 1e3,
            np.nan,
        )
        for aperture_field, active_mask in zip(aperture_fields, aperture_masks)
    ])
    vmin = float(np.nanmin(height_maps_mm))
    vmax = float(np.nanmax(height_maps_mm))
    fig, axes = plt.subplots(
        1,
        len(apertures),
        figsize=(6.0 * len(apertures), 5.2),
        constrained_layout=True,
        squeeze=False,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    radius_mm = 0.5 * diameter_m * 1e3
    for ax, aperture, height_map_mm in zip(
        axes[0],
        apertures,
        height_maps_mm,
    ):
        image = ax.pcolormesh(
            aperture.uu_mm,
            aperture.vv_mm,
            height_map_mm,
            shading="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.plot(
            radius_mm * np.cos(theta),
            radius_mm * np.sin(theta),
            color="black",
            linewidth=1.0,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{aperture.plane.name} hologram height")
        ax.set_xlabel("u [mm]")
        ax.set_ylabel("v [mm]")
        fig.colorbar(image, ax=ax, label="height [mm]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_print_head_for_occlusion_fraction(fraction, line_z_min_m, line_z_max_m, head_xy_m):
    line_length_m = line_z_max_m - line_z_min_m
    tip_z_m = line_z_max_m - fraction * line_length_m
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


def pressure_with_head(
    points_m,
    apertures,
    aperture_fields,
    aperture_masks,
    propagation_models,
    wavelength_m,
    target_extent_m,
    print_head,
    source_stride,
):
    visibility = np.zeros((len(apertures), points_m.shape[0]), dtype=float)
    diagnostics = []
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)

    for idx, (aperture, aperture_field, active_mask, propagation_model) in enumerate(
        zip(apertures, aperture_fields, aperture_masks, propagation_models)
    ):
        visibility[idx], aperture_diagnostics = sampled_aperture_visibility(
            aperture,
            points_m,
            print_head,
            stride=source_stride,
            element_mask=active_mask,
        )
        diagnostics.append(aperture_diagnostics)
        angular_spectrum_contribution = forward_pressure(
            points_m,
            [aperture],
            [aperture_field],
            wavelength_m,
            target_extent_m=target_extent_m,
            propagation_models=[propagation_model],
        )
        pressure += visibility[idx] * angular_spectrum_contribution

    return pressure, visibility, diagnostics


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_PATH)
    required_keys = {
        "acrylic_density_kg_m3",
        "acrylic_inner_face_positions_m",
        "acrylic_impedance_rayl",
        "acrylic_outer_face_positions_m",
        "acrylic_sound_speed_m_s",
        "acrylic_thickness_m",
        "aperture_fields",
        "free_pressure",
        "frequency_hz",
        "hologram_to_acrylic_offset_m",
        "inner_medium",
        "inner_medium_density_kg_m3",
        "inner_medium_sound_speed_m_s",
        "outer_medium",
        "outer_medium_density_kg_m3",
        "outer_medium_sound_speed_m_s",
        "pzt_count",
        "pzt_diameter_m",
        "pzt_n_side",
        "sound_speed_m_s",
        "target_amp",
        "target_extent_m",
        "x",
        "y",
        "z",
    }
    missing_keys = sorted(required_keys.difference(data.files))
    if missing_keys:
        raise ValueError(
            "Saved baseline data is incomplete. "
            f"Rerun 01_generate_unobstructed_vertical_line.py; missing: {missing_keys}"
        )
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_amp = data["target_amp"]
    free_pressure = data["free_pressure"]
    aperture_fields = data["aperture_fields"]
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    target_extent_m = float(data["target_extent_m"])
    pzt_diameter_m = float(data["pzt_diameter_m"])

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_m = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    target_mask = target_amp > 0.0
    target_points_m = points_m[target_mask]
    line_z_min_m = float(target_points_m[:, 2].min())
    line_z_max_m = float(target_points_m[:, 2].max())
    head_xy_m = np.array(((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0))

    wavelength_m = sound_speed_m_s / frequency_hz
    apertures = make_actuators(data)
    propagation_models = make_propagation_models(data, apertures)
    acrylic_outer_faces_m, acrylic_inner_faces_m = validate_acrylic_face_positions(
        data,
        apertures,
        propagation_models,
    )
    if aperture_fields.shape[0] != len(apertures):
        raise ValueError("Saved field count does not match the saved actuator geometry.")
    aperture_masks = np.asarray([
        circular_aperture_mask(aperture, pzt_diameter_m)
        for aperture in apertures
    ])
    if "aperture_masks" in data.files:
        saved_masks = data["aperture_masks"].astype(bool)
        if saved_masks.shape != aperture_masks.shape or not np.array_equal(
            saved_masks,
            aperture_masks,
        ):
            raise ValueError("Saved aperture masks do not match the saved dimensions.")
    aperture_fields = np.asarray([
        field * active_mask
        for field, active_mask in zip(aperture_fields, aperture_masks)
    ])
    expected_field_shape = (len(apertures),) + apertures[0].shape
    if aperture_fields.shape != expected_field_shape:
        raise ValueError(
            f"Saved aperture fields have shape {aperture_fields.shape}; "
            f"expected {expected_field_shape}."
        )
    if free_pressure.shape != target_amp.shape:
        raise ValueError("Saved free pressure and target field shapes do not match.")
    visibility_source_stride = max(
        1,
        int(np.ceil(apertures[0].n_side / VISIBILITY_SAMPLES_PER_SIDE)),
    )
    free_amp = np.abs(free_pressure)
    plot_hologram_heights(
        apertures,
        aperture_fields,
        aperture_masks,
        frequency_hz,
        sound_speed_m_s,
        pzt_diameter_m,
        HEIGHT_PLOT_PATH,
    )

    occlusion_case_config = [
        # Set enabled=False, or comment out a row, to run fewer cases faster.
        {"fraction": .8, "label": "100% occluded", "enabled": True},
        {"fraction": 0.6, "label": "75% occluded", "enabled": True},
        {"fraction": 0.4, "label": "50% occluded", "enabled": True},
        {"fraction": 0.2, "label": "25% occluded", "enabled": True},
    ]
    occlusion_cases = [
        (case["fraction"], case["label"])
        for case in occlusion_case_config
        if case["enabled"]
    ]
    occluded_results = []
    for fraction, label in occlusion_cases:
        print_head = make_print_head_for_occlusion_fraction(
            fraction,
            line_z_min_m=line_z_min_m,
            line_z_max_m=line_z_max_m,
            head_xy_m=head_xy_m,
        )
        pressure, visibility, diagnostics = pressure_with_head(
            points_m,
            apertures,
            aperture_fields,
            aperture_masks,
            propagation_models,
            wavelength_m,
            target_extent_m,
            print_head,
            visibility_source_stride,
        )
        occluded_results.append(
            (fraction, label, print_head, np.abs(pressure), visibility, diagnostics)
        )

    plot_occlusion_slices(
        x=x,
        y=y,
        z=z,
        free_amp=free_amp,
        target_amp=target_amp,
        occluded_results=occluded_results,
        output_path=OUTPUT_PATH,
    )

    print("Rendered occlusion slices from saved unobstructed hologram")
    print(f"Loaded data from {DATA_PATH}")
    print(f"Saved hologram height plot to {HEIGHT_PLOT_PATH}")
    print("Head radius [mm]:", HEAD_RADIUS_M * 1e3)
    print(
        "Hologram-to-acrylic offset [mm]:",
        float(data["hologram_to_acrylic_offset_m"]) * 1e3,
    )
    print("Acrylic thickness [mm]:", float(data["acrylic_thickness_m"]) * 1e3)
    print(
        "Acrylic impedance [MRayl]:",
        float(data["acrylic_impedance_rayl"]) / 1e6,
    )
    print("Outer / inner media:", str(data["outer_medium"]), "/", str(data["inner_medium"]))
    for aperture, outer_face_m, inner_face_m in zip(
        apertures,
        acrylic_outer_faces_m,
        acrylic_inner_faces_m,
    ):
        print(f"{aperture.plane.name} hologram [mm]:", aperture.plane.position_m * 1e3)
        print(f"{aperture.plane.name} acrylic outer face [mm]:", outer_face_m * 1e3)
        print(f"{aperture.plane.name} acrylic inner face [mm]:", inner_face_m * 1e3)
    print("Slice below interface [mm]:", SLICE_BELOW_INTERFACE_M * 1e3)
    source_count = int(np.count_nonzero(
        aperture_masks[0][::visibility_source_stride, ::visibility_source_stride]
    ))
    print("Visibility source samples per circular PZT:", source_count)
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    for _, label, print_head, amp, visibility, diagnostics in occluded_results:
        attenuation = amp / np.maximum(free_amp, 1e-12)
        print(label)
        print("  head bottom z [mm]:", float(print_head.tip_center_m[2] * 1e3))
        print("  plot slice z [mm]:", float((print_head.tip_center_m[2] - SLICE_BELOW_INTERFACE_M) * 1e3))
        print("  mean target |p|:", float(amp[target_mask].mean()))
        print("  mean target attenuation:", float(attenuation[target_mask].mean()))
        print("  PZT A shell shadow:", float(diagnostics[0]["shell_shadow"][target_mask].mean()))
        print("  PZT B shell shadow:", float(diagnostics[1]["shell_shadow"][target_mask].mean()))
    print(f"Saved plot to {OUTPUT_PATH}")


def plot_occlusion_slices(x, y, z, free_amp, target_amp, occluded_results, output_path):
    free_grid = free_amp.reshape((x.size, y.size, z.size))
    target_grid = target_amp.reshape((x.size, y.size, z.size))
    slice_indices = [
        int(np.argmin(np.abs(z - (print_head.tip_center_m[2] - SLICE_BELOW_INTERFACE_M))))
        for _, _, print_head, _, _, _ in occluded_results
    ]
    vmax = max(
        np.percentile(free_grid[:, :, slice_idx], 99.5)
        for slice_idx in slice_indices
    )
    vmax = max(
        vmax,
        max(
            np.percentile(result[3].reshape((x.size, y.size, z.size))[:, :, slice_idx], 99.5)
            for result, slice_idx in zip(occluded_results, slice_indices)
        ),
    )

    fig, axes = plt.subplots(
        2,
        len(occluded_results),
        figsize=(4.1 * len(occluded_results), 8.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = (x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3)

    last_im = None
    for col, ((_, label, print_head, amp, _, _), slice_idx) in enumerate(
        zip(occluded_results, slice_indices)
    ):
        occluded_grid = amp.reshape((x.size, y.size, z.size))
        slice_z_m = z[slice_idx]
        for row, (title_prefix, grid) in enumerate(
            (("Free field", free_grid), ("With head", occluded_grid))
        ):
            ax = axes[row, col]
            last_im = ax.imshow(
                grid[:, :, slice_idx].T,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                aspect="equal",
            )
            target_slice = target_grid[:, :, slice_idx].T
            if np.any(target_slice > 0.0):
                ax.contour(
                    x * 1e3,
                    y * 1e3,
                    target_slice,
                    levels=(0.5,),
                    colors="cyan",
                    linewidths=1.0,
                )
            draw_bottom_circle(ax, print_head)
            ax.set_title(f"{title_prefix}\n{label}, z={slice_z_m * 1e3:.1f} mm")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

    fig.colorbar(last_im, ax=axes, shrink=0.78, label="|p|")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def draw_bottom_circle(ax, print_head):
    theta = np.linspace(0.0, 2.0 * np.pi, 64)
    ax.plot(
        (print_head.tip_center_m[0] + print_head.radius_m * np.cos(theta)) * 1e3,
        (print_head.tip_center_m[1] + print_head.radius_m * np.sin(theta)) * 1e3,
        color="black",
        linewidth=1.0,
        alpha=0.8,
    )


if __name__ == "__main__":
    main()
