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
    make_hologram_aperture,
)
from occlusion_lib import (  # noqa: E402
    ShellAndBottomPrintHead,
    aperture_element_positions,
    ray_transmission_from_source,
)


DATA_PATH = THIS_DIR / "unobstructed_vertical_line_hologram.npz"
OUTPUT_PATH = THIS_DIR / "saved_hologram_occlusion_slices.png"

# Your current print-head parameters. Change these here for occlusion studies.
HEAD_RADIUS_M = 15e-3
HEAD_LENGTH_M = 45e-3
SHELL_SOFT_EDGE_M = 1.0e-3
SHELL_TRANSMISSION = 0.0
BOTTOM_SOFT_EDGE_M = 0.5e-3
BOTTOM_TRANSMISSION = 0.04
BOTTOM_PRESSURE_REFLECTION = -0.98 + 0.0j
SLICE_BELOW_INTERFACE_M = 0.5e-3
VISIBILITY_SOURCE_STRIDE = 4
DIRECT_FIELD_CHUNK_SIZE = 2048


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


def direct_sampled_pressure(
    points_m,
    apertures,
    aperture_fields,
    wavelength_m,
    print_head=None,
    source_stride=VISIBILITY_SOURCE_STRIDE,
    chunk_size=DIRECT_FIELD_CHUNK_SIZE,
):
    k = 2.0 * np.pi / wavelength_m
    visibility = np.zeros((len(apertures), points_m.shape[0]), dtype=float)
    diagnostics = []
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)

    for idx, (aperture, aperture_field) in enumerate(zip(apertures, aperture_fields)):
        sources_m = aperture_element_positions(aperture, stride=source_stride)
        source_values = aperture_field[::source_stride, ::source_stride].ravel()
        source_area_m2 = aperture.dx**2 * source_stride**2
        aperture_pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
        shell_shadow = np.zeros(points_m.shape[0], dtype=float)
        bottom_shadow = np.zeros(points_m.shape[0], dtype=float)

        for start in range(0, points_m.shape[0], chunk_size):
            end = min(start + chunk_size, points_m.shape[0])
            chunk = points_m[start:end]
            chunk_pressure = np.zeros(chunk.shape[0], dtype=np.complex128)
            chunk_visibility = np.zeros(chunk.shape[0], dtype=float)
            chunk_shell_shadow = np.zeros(chunk.shape[0], dtype=float)
            chunk_bottom_shadow = np.zeros(chunk.shape[0], dtype=float)

            for source_m, source_value in zip(sources_m, source_values):
                ray = chunk - source_m[None, :]
                distance_m = np.linalg.norm(ray, axis=1)
                forward_obliquity = np.maximum(
                    (ray @ aperture.plane.normal) / np.maximum(distance_m, 1e-12),
                    0.0,
                )
                if print_head is None:
                    transmission = np.ones(chunk.shape[0], dtype=float)
                    source_shell_shadow = np.zeros(chunk.shape[0], dtype=float)
                    source_bottom_shadow = np.zeros(chunk.shape[0], dtype=float)
                else:
                    transmission, source_diagnostics = ray_transmission_from_source(
                        source_m,
                        chunk,
                        print_head,
                    )
                    source_shell_shadow = source_diagnostics["shell_shadow"]
                    source_bottom_shadow = source_diagnostics["bottom_shadow"]

                kernel = (
                    transmission
                    * forward_obliquity
                    * np.exp(1j * k * distance_m)
                    / np.maximum(distance_m, 1e-6)
                )
                chunk_pressure += source_value * kernel
                chunk_visibility += transmission
                chunk_shell_shadow += source_shell_shadow
                chunk_bottom_shadow += source_bottom_shadow

            aperture_pressure[start:end] = source_area_m2 * chunk_pressure
            visibility[idx, start:end] = chunk_visibility / sources_m.shape[0]
            shell_shadow[start:end] = chunk_shell_shadow / sources_m.shape[0]
            bottom_shadow[start:end] = chunk_bottom_shadow / sources_m.shape[0]

        pressure += aperture_pressure
        diagnostics.append(
            {
                "shell_shadow": shell_shadow,
                "bottom_shadow": bottom_shadow,
                "source_count": sources_m.shape[0],
            }
        )
    return pressure, visibility, diagnostics


def pressure_with_head(
    points_m,
    apertures,
    aperture_fields,
    wavelength_m,
    print_head,
    direct_to_saved_scale,
):
    pressure, visibility, diagnostics = direct_sampled_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        print_head=print_head,
    )
    return direct_to_saved_scale * pressure, visibility, diagnostics


def main():
    data = np.load(DATA_PATH)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_amp = data["target_amp"]
    free_pressure = data["free_pressure"]
    aperture_fields = data["aperture_fields"]
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    target_extent_m = float(data["target_extent_m"])

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    points_m = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))
    target_mask = target_amp > 0.0
    target_points_m = points_m[target_mask]
    line_z_min_m = float(target_points_m[:, 2].min())
    line_z_max_m = float(target_points_m[:, 2].max())
    head_xy_m = np.array(((x[0] + x[-1]) / 2.0, (y[0] + y[-1]) / 2.0))

    wavelength_m = sound_speed_m_s / frequency_hz
    apertures = make_two_usual_actuators()
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
            head_xy_m=head_xy_m,
        )
        pressure, visibility, diagnostics = pressure_with_head(
            points_m,
            apertures,
            aperture_fields,
            wavelength_m,
            target_extent_m,
            print_head,
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
    print("Head radius [mm]:", HEAD_RADIUS_M * 1e3)
    print("Slice below interface [mm]:", SLICE_BELOW_INTERFACE_M * 1e3)
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
