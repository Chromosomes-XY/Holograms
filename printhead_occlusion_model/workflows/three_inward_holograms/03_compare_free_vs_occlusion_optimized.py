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
    evaluate_amplitude_loss,
    forward_pressure,
    initialize_from_focus_points,
    make_hologram_aperture,
    make_optimization_samples,
    optimize_phase_adam,
)
from occlusion_lib import ShellAndBottomPrintHead, sampled_aperture_visibility  # noqa: E402


RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "occlusion_optimized_vertical_line_hologram.npz"
OUTPUT_PATH = RESULTS_DIR / "free_vs_occlusion_optimized_slices.png"

SLICE_BELOW_INTERFACE_M = 0.5e-3
BASELINE_N_ITER = 80
BASELINE_LEARNING_RATE = 0.02
MAX_BACKGROUND_SAMPLES = 1800
PLOT_LEVEL_CONFIG = [
    {"index": 0, "enabled": True},
    {"index": 3, "enabled": True},
    {"index": 6, "enabled": True},
    {"index": 9, "enabled": True},
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


def apply_circular_apertures(apertures, aperture_fields, pzt_diameter_m):
    return [
        field * circular_aperture_mask(aperture, pzt_diameter_m)
        for aperture, field in zip(apertures, aperture_fields)
    ]


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
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    diagnostics = []
    for aperture, aperture_field in zip(apertures, aperture_fields):
        visibility, aperture_diagnostics = sampled_aperture_visibility(
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
        contribution = forward_pressure(
            points_m,
            [aperture],
            [active_aperture_field],
            wavelength_m,
            target_extent_m=target_extent_m,
            distance_bin_m=distance_bin_m,
        )
        pressure += visibility * contribution
    return pressure, diagnostics


def optimize_free_baseline(
    points_m,
    target_amp,
    apertures,
    wavelength_m,
    target_extent_m,
    distance_bin_m,
    pzt_diameter_m,
):
    _, sample_points_m, desired_amp, loss_weights, target_count = make_optimization_samples(
        points_m,
        target_amp,
        max_background=MAX_BACKGROUND_SAMPLES,
    )
    initial_fields = initialize_from_focus_points(
        apertures,
        sample_points_m[:target_count],
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=distance_bin_m,
    )
    _, initial_amp, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        initial_fields,
        desired_amp,
        loss_weights,
        pressure_scale=1.0,
        wavelength_m=wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=distance_bin_m,
    )
    pressure_scale = float(initial_amp[:target_count].mean())
    baseline_fields, history = optimize_phase_adam(
        sample_points_m,
        apertures,
        desired_amp,
        loss_weights,
        pressure_scale,
        target_count,
        wavelength_m,
        target_extent_m,
        n_iter=BASELINE_N_ITER,
        learning_rate=BASELINE_LEARNING_RATE,
        initial_aperture_fields=initial_fields,
        distance_bin_m=distance_bin_m,
    )
    return apply_circular_apertures(apertures, baseline_fields, pzt_diameter_m), history


def main():
    data = np.load(DATA_PATH)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_amp = data["target_amp"]
    occlusion_optimized_fields = data["aperture_fields"]
    occlusion_fractions = data["occlusion_fractions"]
    head_tip_positions_m = data["head_tip_positions_m"]
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    target_extent_m = float(data["target_extent_m"])
    distance_bin_m = float(data["distance_bin_m"])
    visibility_source_stride = int(data["visibility_source_stride"])
    pzt_diameter_m = float(data["pzt_diameter_m"])

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_m = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    target_mask = target_amp > 0.0
    wavelength_m = sound_speed_m_s / frequency_hz
    apertures = make_three_actuators(data)

    print("Optimizing free-field baseline for comparison...")
    free_optimized_fields, free_history = optimize_free_baseline(
        points_m,
        target_amp,
        apertures,
        wavelength_m,
        target_extent_m,
        distance_bin_m,
        pzt_diameter_m,
    )

    enabled_indices = [
        case["index"]
        for case in PLOT_LEVEL_CONFIG
        if case["enabled"] and case["index"] < len(occlusion_fractions)
    ]
    comparison_results = []
    for level_index in enabled_indices:
        print_head = make_print_head(head_tip_positions_m[level_index], data)
        free_pressure, free_diagnostics = pressure_with_head(
            points_m,
            apertures,
            free_optimized_fields,
            wavelength_m,
            target_extent_m,
            distance_bin_m,
            visibility_source_stride,
            pzt_diameter_m,
            print_head,
        )
        aggregate_pressure, aggregate_diagnostics = pressure_with_head(
            points_m,
            apertures,
            occlusion_optimized_fields,
            wavelength_m,
            target_extent_m,
            distance_bin_m,
            visibility_source_stride,
            pzt_diameter_m,
            print_head,
        )
        comparison_results.append(
            {
                "fraction": float(occlusion_fractions[level_index]),
                "print_head": print_head,
                "free_amp": np.abs(free_pressure),
                "aggregate_amp": np.abs(aggregate_pressure),
                "free_diagnostics": free_diagnostics,
                "aggregate_diagnostics": aggregate_diagnostics,
            }
        )

    plot_comparison(x, y, z, target_amp, comparison_results, OUTPUT_PATH)

    print("Rendered free-optimized versus occlusion-optimized comparison")
    print("Free baseline final loss:", free_history["loss"][-1])
    for result in comparison_results:
        free_amp = result["free_amp"]
        aggregate_amp = result["aggregate_amp"]
        print(f"{result['fraction']:.2f} occlusion")
        print("  free-optimized mean target |p|:", float(free_amp[target_mask].mean()))
        print("  occlusion-optimized mean target |p|:", float(aggregate_amp[target_mask].mean()))
    print(f"Saved plot to {OUTPUT_PATH}")


def plot_comparison(x, y, z, target_amp, comparison_results, output_path):
    if not comparison_results:
        raise ValueError("No occlusion levels are enabled for comparison.")

    target_grid = target_amp.reshape((x.size, y.size, z.size))
    slice_indices = [
        int(
            np.argmin(
                np.abs(
                    z
                    - np.clip(
                        result["print_head"].tip_center_m[2] - SLICE_BELOW_INTERFACE_M,
                        z[0],
                        z[-1],
                    )
                )
            )
        )
        for result in comparison_results
    ]
    vmax = 0.0
    for result, slice_idx in zip(comparison_results, slice_indices):
        for key in ("free_amp", "aggregate_amp"):
            grid = result[key].reshape((x.size, y.size, z.size))
            vmax = max(vmax, np.percentile(grid[:, :, slice_idx], 99.5))

    fig, axes = plt.subplots(
        2,
        len(comparison_results),
        figsize=(4.1 * len(comparison_results), 8.0),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    extent = (x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3)
    rows = (
        ("Free-optimized", "free_amp"),
        ("Occlusion-optimized", "aggregate_amp"),
    )
    last_im = None
    for col, (result, slice_idx) in enumerate(zip(comparison_results, slice_indices)):
        target_slice = target_grid[:, :, slice_idx].T
        for row, (row_label, amp_key) in enumerate(rows):
            grid = result[amp_key].reshape((x.size, y.size, z.size))
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
            if np.any(target_slice > 0.0):
                ax.contour(
                    x * 1e3,
                    y * 1e3,
                    target_slice,
                    levels=(0.5,),
                    colors="cyan",
                    linewidths=1.0,
                )
            draw_bottom_circle(ax, result["print_head"])
            ax.set_title(
                f"{row_label}\n"
                f"{result['fraction']:.2f} occlusion, z={z[slice_idx] * 1e3:.1f} mm"
            )
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
