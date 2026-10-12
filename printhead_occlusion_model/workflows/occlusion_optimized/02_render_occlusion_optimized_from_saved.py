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
SLICE_BELOW_INTERFACE_M = 0.6e-3
PLOT_LEVEL_CONFIG = [
    # Set enabled=False, or comment out a row, to render fewer cases faster.
    {"index": 0, "enabled": True},
    {"index": 33, "enabled": True},
    {"index": 66, "enabled": True},
    {"index": 99, "enabled": True},
]


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


def make_print_head(tip_m, data):
    return ShellAndBottomPrintHead.from_tip_position(
        tip_m=tip_m,
        axis=(0.0, 0.0, 1.0),
        radius_m=float(data["head_radius_m"]),
        length_m=float(data["head_length_m"]),
        shell_soft_edge_m=float(data["shell_soft_edge_m"]),
        shell_transmission=float(data["shell_transmission"]),
        bottom_soft_edge_m=float(data["bottom_soft_edge_m"]),
        bottom_transmission=float(data["bottom_transmission"]),
        bottom_pressure_reflection=complex(data["bottom_pressure_reflection"]),
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


def main():
    data = np.load(DATA_PATH)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_amp = data["target_amp"]
    free_pressure = data["free_pressure"]
    aperture_fields = data["aperture_fields"]
    occlusion_fractions = data["occlusion_fractions"]
    head_tip_positions_m = data["head_tip_positions_m"]
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    target_extent_m = float(data["target_extent_m"])
    distance_bin_m = float(data["distance_bin_m"])
    visibility_source_stride = int(data["visibility_source_stride"])
    pzt_n_side = int(data["pzt_n_side"])
    pzt_diameter_m = float(data["pzt_diameter_m"])

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_m = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    target_mask = target_amp > 0.0

    wavelength_m = sound_speed_m_s / frequency_hz
    apertures = make_three_actuators(data)
    free_amp = np.abs(free_pressure)

    enabled_indices = [
        case["index"]
        for case in PLOT_LEVEL_CONFIG
        if case["enabled"] and case["index"] < len(occlusion_fractions)
    ]
    occluded_results = []
    for level_index in enabled_indices:
        print_head = make_print_head(head_tip_positions_m[level_index], data)
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
                np.abs(pressure),
                visibility,
                diagnostics,
            )
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

    print("Rendered aggregate-optimized hologram occlusion slices")
    print(f"Loaded data from {DATA_PATH}")
    print("Slice below interface [mm]:", SLICE_BELOW_INTERFACE_M * 1e3)
    print("Visibility source samples per PZT:", (pzt_n_side // visibility_source_stride) ** 2)
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    for _, label, print_head, amp, visibility, diagnostics in occluded_results:
        attenuation = amp / np.maximum(free_amp, 1e-12)
        print(label)
        print("  head bottom z [mm]:", float(print_head.tip_center_m[2] * 1e3))
        print("  plot slice z [mm]:", float((print_head.tip_center_m[2] - SLICE_BELOW_INTERFACE_M) * 1e3))
        print("  mean target |p|:", float(amp[target_mask].mean()))
        print("  mean target attenuation:", float(attenuation[target_mask].mean()))
        for aperture, aperture_diagnostics in zip(apertures, diagnostics):
            print(
                f"  {aperture.plane.name} shell shadow:",
                float(aperture_diagnostics["shell_shadow"][target_mask].mean()),
            )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saved plot to {OUTPUT_PATH}")


def plot_occlusion_slices(x, y, z, free_amp, target_amp, occluded_results, output_path):
    if not occluded_results:
        raise ValueError("No occlusion levels are enabled for plotting.")

    free_grid = free_amp.reshape((x.size, y.size, z.size))
    target_grid = target_amp.reshape((x.size, y.size, z.size))
    slice_indices = [
        int(np.argmin(np.abs(z - np.clip(print_head.tip_center_m[2] - SLICE_BELOW_INTERFACE_M, z[0], z[-1]))))
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
