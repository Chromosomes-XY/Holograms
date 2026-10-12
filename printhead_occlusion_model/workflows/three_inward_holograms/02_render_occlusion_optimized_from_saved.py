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
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))
sys.path.insert(0, str(PRINTHEAD_DIR / "lib"))

from hologram3d_lib import (  # noqa: E402
    OrientedPlane,
    forward_pressure,
    make_hologram_aperture,
)
from occlusion_lib import ShellAndBottomPrintHead, sampled_aperture_visibility  # noqa: E402


RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "occlusion_optimized_vertical_line_hologram.npz"
OUTPUT_PATH = RESULTS_DIR / "occlusion_optimized_saved_slices.png"

# Display-only controls. The physical parameters are loaded from DATA_PATH.
PLOT_Z_MIN_M = -5e-3
PLOT_Z_MAX_M = 20e-3
PLOT_SLICE_BELOW_HEAD_M = 0.6e-3
PLOT_VISIBILITY_SOURCE_STRIDE = 32
PLOT_LEVEL_CONFIG = [
    # Set enabled=False, or comment out a row, to render fewer cases faster.
    {"index": 0, "enabled": True},
    {"index": 3, "enabled": True},
    {"index": 6, "enabled": True},
    {"index": 9, "enabled": True},
]


def load_optimizer_config():
    optimizer_path = THIS_DIR / "01_optimize_for_occlusion_levels.py"
    spec = importlib.util.spec_from_file_location("three_inward_optimizer_config", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_three_actuators(data):
    pzt_n_side = int(data["pzt_n_side"])
    pzt_width_m = float(data["pzt_width_m"])
    pzt_positions_m = data["pzt_positions_m"]
    pzt_normals = data["pzt_normals"]
    pzt_tangent_hints = data["pzt_tangent_hints"]
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


def circular_aperture_mask(aperture, pzt_diameter_m):
    return (aperture.uu_mm**2 + aperture.vv_mm**2) <= (0.5 * pzt_diameter_m * 1e3) ** 2


def make_print_head(tip_m, data, config=None):
    if "head_radius_m" in data.files:
        radius_m = float(data["head_radius_m"])
        length_m = float(data["head_length_m"])
        shell_soft_edge_m = float(data["shell_soft_edge_m"])
        shell_transmission = float(data["shell_transmission"])
        bottom_soft_edge_m = float(data["bottom_soft_edge_m"])
        bottom_transmission = float(data["bottom_transmission"])
        bottom_pressure_reflection = complex(data["bottom_pressure_reflection"])
    else:
        if config is None:
            config = load_optimizer_config()
        radius_m = config.HEAD_RADIUS_M
        length_m = config.HEAD_LENGTH_M
        shell_soft_edge_m = config.SHELL_SOFT_EDGE_M
        shell_transmission = config.SHELL_TRANSMISSION
        bottom_soft_edge_m = config.BOTTOM_SOFT_EDGE_M
        bottom_transmission = config.BOTTOM_TRANSMISSION
        bottom_pressure_reflection = config.BOTTOM_PRESSURE_REFLECTION

    return ShellAndBottomPrintHead.from_tip_position(
        tip_m=tip_m,
        axis=(0.0, 0.0, 1.0),
        radius_m=radius_m,
        length_m=length_m,
        shell_soft_edge_m=shell_soft_edge_m,
        shell_transmission=shell_transmission,
        bottom_soft_edge_m=bottom_soft_edge_m,
        bottom_transmission=bottom_transmission,
        bottom_pressure_reflection=bottom_pressure_reflection,
    )


def pressure_with_head(
    points_m,
    apertures,
    aperture_fields,
    wavelength_m,
    target_extent_m,
    distance_bin_m,
    visibility_source_stride,
    pzt_diameter_m,
    print_head,
):
    visibility = np.zeros((len(apertures), points_m.shape[0]), dtype=float)
    diagnostics = []
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)

    for idx, (aperture, aperture_field) in enumerate(zip(apertures, aperture_fields)):
        visibility[idx], aperture_diagnostics = sampled_aperture_visibility(
            aperture,
            points_m,
            print_head,
            stride=visibility_source_stride,
        )
        diagnostics.append(aperture_diagnostics)
        active_aperture_field = aperture_field * circular_aperture_mask(
            aperture,
            pzt_diameter_m,
        )
        angular_spectrum_contribution = forward_pressure(
            points_m,
            [aperture],
            [active_aperture_field],
            wavelength_m,
            target_extent_m=target_extent_m,
            distance_bin_m=distance_bin_m,
        )
        pressure += visibility[idx] * angular_spectrum_contribution

    return pressure, visibility, diagnostics


def render_saved_hologram(data_path, output_path, source_label):
    data = np.load(data_path)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_grid = data["target_amp"].reshape((x.size, y.size, z.size))
    free_grid = np.abs(data["free_pressure"]).reshape((x.size, y.size, z.size))
    aperture_fields = data["aperture_fields"]
    config = None
    if "occlusion_fractions" in data.files and "head_tip_positions_m" in data.files:
        occlusion_fractions = data["occlusion_fractions"]
        head_tip_positions_m = data["head_tip_positions_m"]
    else:
        config = load_optimizer_config()
        occlusion_fractions = np.asarray(config.OCCLUSION_FRACTIONS)
        head_xy_m = np.array(((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0))
        head_tip_positions_m = np.asarray(
            [
                config.make_print_head_for_occlusion_fraction(fraction, z[0], z[-1], head_xy_m).tip_center_m
                for fraction in occlusion_fractions
            ]
        )
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    target_extent_m = float(data["target_extent_m"])
    distance_bin_m = float(data["distance_bin_m"])
    visibility_source_stride = PLOT_VISIBILITY_SOURCE_STRIDE
    pzt_n_side = int(data["pzt_n_side"])
    pzt_diameter_m = float(data["pzt_diameter_m"])

    wavelength_m = sound_speed_m_s / frequency_hz
    apertures = make_three_actuators(data)

    enabled_indices = [
        case["index"]
        for case in PLOT_LEVEL_CONFIG
        if case["enabled"] and case["index"] < len(occlusion_fractions)
    ]
    occluded_results = []
    for level_index in enabled_indices:
        print_head = make_print_head(head_tip_positions_m[level_index], data, config=config)
        z_indices = plot_z_indices(z, print_head)
        points_m = make_xy_z_points(x, y, z[z_indices])
        pressure, visibility, diagnostics = pressure_with_head(
            points_m,
            apertures,
            aperture_fields,
            wavelength_m,
            target_extent_m,
            distance_bin_m,
            visibility_source_stride,
            pzt_diameter_m,
            print_head,
        )
        label = f"{occlusion_fractions[level_index]:.2f} occlusion"
        occluded_results.append(
            (
                float(occlusion_fractions[level_index]),
                label,
                print_head,
                np.abs(pressure).reshape((x.size, y.size, z_indices.size)),
                visibility,
                diagnostics,
                z_indices,
            )
        )

    plot_occlusion_slices(
        x=x,
        y=y,
        z=z,
        free_grid=free_grid,
        target_grid=target_grid,
        occluded_results=occluded_results,
        output_path=output_path,
    )

    print(f"Rendered {source_label} hologram occlusion slices")
    print(f"Loaded data from {data_path}")
    print("Plot z range [mm]:", f"{PLOT_Z_MIN_M * 1e3:.3f} to {PLOT_Z_MAX_M * 1e3:.3f}")
    print("Plot slice below head [mm]:", PLOT_SLICE_BELOW_HEAD_M * 1e3)
    print("Plot visibility source stride:", PLOT_VISIBILITY_SOURCE_STRIDE)
    print("Visibility source samples per PZT:", (pzt_n_side // visibility_source_stride) ** 2)
    for _, label, print_head, amp_z_range, visibility, diagnostics, z_indices in occluded_results:
        target_mask = np.max(target_grid[:, :, z_indices], axis=2).ravel() > 0.0
        free_slice = np.max(free_grid[:, :, z_indices], axis=2).ravel()
        amp_flat = np.max(amp_z_range, axis=2).ravel()
        attenuation = amp_flat / np.maximum(free_slice, 1e-12)
        print(label)
        print("  head bottom z [mm]:", float(print_head.tip_center_m[2] * 1e3))
        print("  plotted z [mm]:", f"{z[z_indices[0]] * 1e3:.3f} to {z[z_indices[-1]] * 1e3:.3f}")
        if np.any(target_mask):
            print("  mean target |p| free:", float(free_slice[target_mask].mean()))
            print("  mean target |p|:", float(amp_flat[target_mask].mean()))
            print("  mean target attenuation:", float(attenuation[target_mask].mean()))
        else:
            print("  mean target |p|: no demanded target points on this slice")
        for aperture, aperture_diagnostics in zip(apertures, diagnostics):
            if np.any(target_mask):
                shell_shadow = np.max(
                    aperture_diagnostics["shell_shadow"].reshape((x.size, y.size, z_indices.size)),
                    axis=2,
                ).ravel()
                print(
                    f"  {aperture.plane.name} shell shadow:",
                    float(shell_shadow[target_mask].mean()),
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved plot to {output_path}")


def main():
    render_saved_hologram(DATA_PATH, OUTPUT_PATH, "occlusion-optimized")


def make_xy_z_points(x, y, z_values_m):
    xv, yv, zv = np.meshgrid(x, y, z_values_m, indexing="ij")
    return np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))


def plot_z_indices(z, print_head):
    slice_z_m = print_head.tip_center_m[2] - PLOT_SLICE_BELOW_HEAD_M
    slice_z_m = np.clip(slice_z_m, PLOT_Z_MIN_M, PLOT_Z_MAX_M)
    slice_z_m = np.clip(slice_z_m, z[0], z[-1])
    return np.array([int(np.argmin(np.abs(z - slice_z_m)))])


def plot_occlusion_slices(x, y, z, free_grid, target_grid, occluded_results, output_path):
    if not occluded_results:
        raise ValueError("No occlusion levels are enabled for plotting.")

    z_indices_by_result = [result[6] for result in occluded_results]
    vmax = max(
        np.percentile(np.max(free_grid[:, :, z_indices], axis=2), 99.5)
        for z_indices in z_indices_by_result
    )
    vmax = max(
        vmax,
        max(
            np.percentile(np.max(result[3], axis=2), 99.5)
            for result in occluded_results
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
    for col, ((_, label, print_head, amp_z_range, _, _, z_indices)) in enumerate(occluded_results):
        z_label = f"{z[z_indices[0]] * 1e3:.1f}..{z[z_indices[-1]] * 1e3:.1f}"
        for row, (title_prefix, slice_data) in enumerate(
            (
                ("Free field", np.max(free_grid[:, :, z_indices], axis=2)),
                ("With head", np.max(amp_z_range, axis=2)),
            )
        ):
            ax = axes[row, col]
            last_im = ax.imshow(
                slice_data.T,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
                aspect="equal",
            )
            target_slice = np.max(target_grid[:, :, z_indices], axis=2).T
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
            ax.set_title(f"{title_prefix}\n{label}, z={z_label} mm")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

    fig.colorbar(last_im, ax=axes, shrink=0.78, label="|p|")
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
