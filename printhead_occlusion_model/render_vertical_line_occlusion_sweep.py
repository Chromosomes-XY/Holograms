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
from occlusion_lib import ShellAndBottomPrintHead, aperture_visibility  # noqa: E402


OUTPUT_PATH = THIS_DIR / "vertical_line_occlusion_sweep.png"


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


def make_print_head_for_occlusion_fraction(fraction, line_z_min_m, line_z_max_m):
    line_length_m = line_z_max_m - line_z_min_m
    tip_z_m = line_z_max_m - fraction * line_length_m
    return ShellAndBottomPrintHead.from_tip_position(
        tip_m=(20e-3, 0.0, tip_z_m),
        axis=(0.0, 0.0, 1.0),
        radius_m=15e-3,
        length_m=45e-3,
        shell_soft_edge_m=1.0e-3,
        shell_transmission=0.0,
        bottom_soft_edge_m=0.5e-3,
        bottom_transmission=0.04,
        bottom_pressure_reflection=-0.98 + 0.0j,
    )


def pressure_with_head(points_m, apertures, aperture_fields, wavelength_m, target_extent_m, print_head):
    visibility, diagnostics = aperture_visibility(apertures, points_m, print_head)
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    for idx, (aperture, aperture_field) in enumerate(zip(apertures, aperture_fields)):
        contribution = forward_pressure(
            points_m,
            [aperture],
            [aperture_field],
            wavelength_m,
            target_extent_m=target_extent_m,
        )
        pressure += visibility[idx] * contribution
    return pressure, visibility, diagnostics


def main():
    frequency_hz = 2.05e6
    sound_speed_m_s = 1500.0
    wavelength_m = sound_speed_m_s / frequency_hz

    x, y, z, points_m = make_volume_grid(
        bounds_m=((0.0, 40e-3), (-20e-3, 20e-3), (-20e-3, 20e-3)),
        n=(49, 49, 49),
    )
    target_extent_m = max(x[-1] - x[0], y[-1] - y[0], z[-1] - z[0])

    line_z_min_m = -15e-3
    line_z_max_m = 15e-3
    vertical_line_m = [
        (np.array((20e-3, 0.0, line_z_min_m)), np.array((20e-3, 0.0, line_z_max_m)))
    ]
    target_amp = make_line_segment_target(points_m, vertical_line_m, tube_radius_m=0.5e-3)
    target_mask = target_amp > 0.0

    apertures = make_two_usual_actuators()
    _, sample_points_m, desired_amp, loss_weights, target_count = make_optimization_samples(
        points_m,
        target_amp,
        max_background=4500,
    )
    if target_count == 0:
        raise ValueError("Vertical line target sampled zero points.")

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

    occlusion_cases = [
        (1.00, "100% occluded"),
        (0.75, "75% occluded"),
        (0.50, "50% occluded"),
        (0.25, "25% occluded"),
    ]
    occluded_results = []
    for fraction, label in occlusion_cases:
        print_head = make_print_head_for_occlusion_fraction(
            fraction,
            line_z_min_m=line_z_min_m,
            line_z_max_m=line_z_max_m,
        )
        pressure, visibility, diagnostics = pressure_with_head(
            points_m,
            apertures,
            aperture_fields,
            wavelength_m,
            target_extent_m,
            print_head,
        )
        amp = np.abs(pressure)
        occluded_results.append((fraction, label, print_head, amp, visibility, diagnostics))

    plot_occlusion_sweep(
        x=x,
        y=y,
        z=z,
        free_amp=free_amp,
        target_amp=target_amp,
        occluded_results=occluded_results,
        output_path=OUTPUT_PATH,
    )

    print("Unobstructed vertical-line hologram")
    print("Target samples:", target_count)
    print("Initial loss:", initial_loss)
    print("Optimized loss:", history["loss"][-1])
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    print("Mean background |p| free:", float(free_amp[~target_mask].mean()))
    for fraction, label, print_head, amp, visibility, diagnostics in occluded_results:
        attenuation = amp / np.maximum(free_amp, 1e-12)
        print(label)
        print("  head bottom z [mm]:", float(print_head.tip_center_m[2] * 1e3))
        print("  mean target |p|:", float(amp[target_mask].mean()))
        print("  mean target attenuation:", float(attenuation[target_mask].mean()))
        print("  PZT A mean target visibility:", float(visibility[0, target_mask].mean()))
        print("  PZT B mean target visibility:", float(visibility[1, target_mask].mean()))
        print("  PZT A shell shadow:", float(diagnostics[0]["shell_shadow"][target_mask].mean()))
        print("  PZT B shell shadow:", float(diagnostics[1]["shell_shadow"][target_mask].mean()))
    print(f"Saved plot to {OUTPUT_PATH}")


def plot_occlusion_sweep(x, y, z, free_amp, target_amp, occluded_results, output_path):
    free_grid = free_amp.reshape((x.size, y.size, z.size))
    target_grid = target_amp.reshape((x.size, y.size, z.size))
    slice_offset_m = 0.5e-3
    slice_indices = [
        int(np.argmin(np.abs(z - (print_head.tip_center_m[2] - slice_offset_m))))
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
    )
    extent = (x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3)

    last_im = None
    for col, ((fraction, label, print_head, amp, _, _), slice_idx) in enumerate(
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
            ax.set_title(
                f"{title_prefix}\n{label}, z={slice_z_m * 1e3:.1f} mm"
            )
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
