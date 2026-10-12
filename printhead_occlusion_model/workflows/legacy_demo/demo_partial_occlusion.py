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
    initialize_from_focus_points,
    make_hologram_aperture,
    make_line_segment_target,
    make_volume_grid,
)
from occlusion_lib import (  # noqa: E402
    ShellAndBottomPrintHead,
    aperture_visibility,
)


OUTPUT_PATH = THIS_DIR / "results" / "partial_occlusion_demo.png"


def make_two_pzt_geometry():
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
        make_hologram_aperture(plane, n_side=64, width_m=20e-3)
        for plane in pzt_planes
    ]


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frequency_hz = 2.05e6
    sound_speed_m_s = 1500.0
    wavelength_m = sound_speed_m_s / frequency_hz

    apertures = make_two_pzt_geometry()
    x, y, z, points_m = make_volume_grid(
        bounds_m=((0.0, 40e-3), (-20e-3, 20e-3), (-20e-3, 20e-3)),
        n=(49, 49, 49),
    )

    line_segments_m = [
        (np.array((20e-3, -15e-3, 0.0)), np.array((20e-3, 15e-3, 0.0))),
        (np.array((20e-3, 0.0, -15e-3)), np.array((20e-3, 0.0, 15e-3))),
    ]
    target_amp = make_line_segment_target(points_m, line_segments_m, tube_radius_m=0.5e-3)
    target_points_m = points_m[target_amp > 0.0]
    target_extent_m = max(x[-1] - x[0], y[-1] - y[0], z[-1] - z[0])

    aperture_fields = initialize_from_focus_points(
        apertures,
        target_points_m,
        wavelength_m,
        target_extent_m=target_extent_m,
    )

    free_pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
    )

    print_head = ShellAndBottomPrintHead.from_tip_position(
        tip_m=(20e-3, 0.0, -18e-3),
        axis=(0.0, 0.0, 1.0),
        radius_m=2.5e-3,
        length_m=36e-3,
        shell_soft_edge_m=1.0e-3,
        shell_transmission=0.05,
        bottom_soft_edge_m=0.5e-3,
        bottom_transmission=0.04,
        bottom_pressure_reflection=-0.98 + 0.0j,
    )
    visibility, diagnostics = aperture_visibility(apertures, points_m, print_head)

    occluded_pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    for idx, (aperture, aperture_field) in enumerate(zip(apertures, aperture_fields)):
        contribution = forward_pressure(
            points_m,
            [aperture],
            [aperture_field],
            wavelength_m,
            target_extent_m=target_extent_m,
        )
        occluded_pressure += visibility[idx] * contribution

    free_amp = np.abs(free_pressure)
    occluded_amp = np.abs(occluded_pressure)
    attenuation = occluded_amp / np.maximum(free_amp, 1e-12)

    target_mask = target_amp > 0.0
    print("Print-head partial occlusion demo")
    print("Shell start / bottom centre [m]:", print_head.shell_start_m)
    print("Shell end [m]:", print_head.shell_end_m)
    print("Head radius [m]:", print_head.radius_m)
    print("Absorbing shell transmission:", print_head.shell_transmission)
    print("Reflective bottom transmission:", print_head.bottom_transmission)
    print("Reflective bottom pressure reflection:", print_head.bottom_pressure_reflection)
    print("Mean target |p| free:", float(free_amp[target_mask].mean()))
    print("Mean target |p| occluded:", float(occluded_amp[target_mask].mean()))
    print("Mean target attenuation:", float(attenuation[target_mask].mean()))
    for idx, aperture in enumerate(apertures):
        print(
            f"{aperture.plane.name} mean target visibility:",
            float(visibility[idx, target_mask].mean()),
        )
        print(
            f"{aperture.plane.name} mean target shell shadow:",
            float(diagnostics[idx]["shell_shadow"][target_mask].mean()),
        )
        print(
            f"{aperture.plane.name} mean target bottom shadow:",
            float(diagnostics[idx]["bottom_shadow"][target_mask].mean()),
        )

    plot_slices(
        x=x,
        y=y,
        z=z,
        free_amp=free_amp.reshape((x.size, y.size, z.size)),
        occluded_amp=occluded_amp.reshape((x.size, y.size, z.size)),
        attenuation=attenuation.reshape((x.size, y.size, z.size)),
        target_amp=target_amp.reshape((x.size, y.size, z.size)),
        output_path=OUTPUT_PATH,
    )
    print(f"Saved plot to {OUTPUT_PATH}")


def plot_slices(x, y, z, free_amp, occluded_amp, attenuation, target_amp, output_path):
    z_idx = int(np.argmin(np.abs(z - 0.0)))
    extent = (x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3)

    vmax = np.percentile(free_amp[:, :, z_idx], 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)

    panels = [
        (free_amp[:, :, z_idx], "Free field |p|", "magma", 0.0, vmax),
        (occluded_amp[:, :, z_idx], "With print-head occlusion |p|", "magma", 0.0, vmax),
        (attenuation[:, :, z_idx], "Occluded / free", "viridis", 0.0, 1.0),
    ]
    for ax, (data, title, cmap, vmin, panel_vmax) in zip(axes, panels):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=panel_vmax,
            aspect="equal",
        )
        contour = target_amp[:, :, z_idx].T
        if np.any(contour > 0.0):
            ax.contour(
                x * 1e3,
                y * 1e3,
                contour,
                levels=(0.5,),
                colors="cyan",
                linewidths=1.0,
            )
        ax.set_title(title)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
