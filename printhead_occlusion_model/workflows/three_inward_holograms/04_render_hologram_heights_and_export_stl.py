from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
REPO_ROOT = PRINTHEAD_DIR.parent
RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "occlusion_optimized_vertical_line_hologram.npz"
HEIGHT_PLOT_PATH = RESULTS_DIR / "hologram_height_arrays.png"
HEIGHT_DATA_PATH = RESULTS_DIR / "hologram_height_arrays.npz"
STL_DIR = RESULTS_DIR / "hologram_stls"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))

from hologram3d_lib import OrientedPlane, make_hologram_aperture  # noqa: E402
from hologramGen import acoustic_params, export_hologram_stl  # noqa: E402


# Match the earlier single-hologram generator material defaults.
C_HOLOGRAM_M_S = 2550.0
BASE_THICKNESS_M = 0.4e-3
FIELD_PERCENTILE = 99.5


def make_actuators(data):
    pzt_n_side = int(data["pzt_n_side"])
    pzt_width_m = float(data["pzt_width_m"])
    pzt_positions_m = data["pzt_positions_m"]
    pzt_normals = data["pzt_normals"]
    pzt_tangent_hints = data["pzt_tangent_hints"]
    planes = [
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
        for plane in planes
    ]


def circular_aperture_mask(aperture, pzt_diameter_m):
    radius_mm = 0.5 * pzt_diameter_m * 1e3
    return aperture.uu_mm**2 + aperture.vv_mm**2 <= radius_mm**2


def phase_to_printable_height_m(phase_rad, frequency_hz, sound_speed_m_s):
    acoustic = acoustic_params(
        frequency_hz=frequency_hz,
        c_medium=sound_speed_m_s,
        c_hologram=C_HOLOGRAM_M_S,
    )
    delta_k = acoustic["km"] - acoustic["kh"]
    if delta_k <= 0.0:
        raise ValueError("Hologram material speed must exceed medium speed for this height mapping.")
    wrapped_phase = np.mod(phase_rad, 2.0 * np.pi)
    return BASE_THICKNESS_M + wrapped_phase / delta_k


def make_height_maps(data, apertures):
    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    pzt_diameter_m = float(data["pzt_diameter_m"])

    height_maps = []
    masks = []
    for aperture, aperture_field in zip(apertures, data["aperture_fields"]):
        mask = circular_aperture_mask(aperture, pzt_diameter_m)
        height_m = phase_to_printable_height_m(
            np.angle(aperture_field),
            frequency_hz=frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
        )
        height_m = np.where(mask, height_m, np.nan)
        height_maps.append(height_m)
        masks.append(mask)
    return np.asarray(height_maps), np.asarray(masks)


def plot_height_arrays(data, apertures, height_maps_m, output_path):
    x = data["x"]
    y = data["y"]
    z = data["z"]
    target_amp = data["target_amp"].reshape((x.size, y.size, z.size))
    pressure_amp = np.abs(data["free_pressure"]).reshape((x.size, y.size, z.size))

    field_threshold = np.percentile(pressure_amp, FIELD_PERCENTILE)
    field_mask = pressure_amp >= field_threshold
    target_mask = target_amp > 0.0
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    fig = plt.figure(figsize=(18.5, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, len(apertures) + 1, width_ratios=[1.3, 1.0, 1.0, 1.0])

    ax_field = fig.add_subplot(gs[0, 0], projection="3d")
    scatter = ax_field.scatter(
        xv[field_mask] * 1e3,
        yv[field_mask] * 1e3,
        zv[field_mask] * 1e3,
        c=pressure_amp[field_mask],
        s=10,
        alpha=0.55,
        cmap="Reds",
    )
    if np.any(target_mask):
        ax_field.scatter(
            xv[target_mask] * 1e3,
            yv[target_mask] * 1e3,
            zv[target_mask] * 1e3,
            c="cyan",
            s=12,
            alpha=0.85,
            marker="x",
        )
    ax_field.set_title("3D field strength")
    ax_field.set_xlabel("x [mm]")
    ax_field.set_ylabel("y [mm]")
    ax_field.set_zlabel("z [mm]")
    ax_field.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax_field.set_ylim(y[0] * 1e3, y[-1] * 1e3)
    ax_field.set_zlim(z[0] * 1e3, z[-1] * 1e3)
    fig.colorbar(scatter, ax=ax_field, shrink=0.78, label="|p|")

    height_mm = height_maps_m * 1e3
    vmin = float(np.nanmin(height_mm))
    vmax = float(np.nanmax(height_mm))
    for idx, (aperture, height_map_mm) in enumerate(zip(apertures, height_mm)):
        ax = fig.add_subplot(gs[0, idx + 1])
        im = ax.pcolormesh(
            aperture.uu_mm,
            aperture.vv_mm,
            height_map_mm,
            shading="nearest",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        draw_aperture_outline(ax, float(data["pzt_diameter_m"]) * 1e3)
        ax.set_aspect("equal")
        ax.set_title(f"{aperture.plane.name} height")
        ax.set_xlabel("u [mm]")
        ax.set_ylabel("v [mm]")
        ax.set_xlim(
            -0.5 * float(data["pzt_width_m"]) * 1e3,
            0.5 * float(data["pzt_width_m"]) * 1e3,
        )
        ax.set_ylim(
            -0.5 * float(data["pzt_width_m"]) * 1e3,
            0.5 * float(data["pzt_width_m"]) * 1e3,
        )
        fig.colorbar(im, ax=ax, shrink=0.95, label="height [mm]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def draw_aperture_outline(ax, diameter_mm):
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    radius_mm = 0.5 * diameter_mm
    ax.plot(radius_mm * np.cos(theta), radius_mm * np.sin(theta), color="black", linewidth=1.0)


def export_height_stls(apertures, height_maps_m):
    STL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (aperture, height_map_m) in enumerate(zip(apertures, height_maps_m)):
        label = chr(ord("a") + idx)
        output_path = STL_DIR / f"pzt_{label}_hologram_height.stl"
        n_facets, nonmanifold_edges = export_hologram_stl(
            height_map_m,
            aperture.dx,
            output_path=output_path,
        )
        results.append((output_path, n_facets, nonmanifold_edges))
    return results


def main():
    data = np.load(DATA_PATH)
    apertures = make_actuators(data)
    height_maps_m, masks = make_height_maps(data, apertures)

    plot_height_arrays(data, apertures, height_maps_m, HEIGHT_PLOT_PATH)
    stl_results = export_height_stls(apertures, height_maps_m)

    np.savez(
        HEIGHT_DATA_PATH,
        height_maps_m=height_maps_m,
        aperture_masks=masks,
        pzt_positions_m=data["pzt_positions_m"],
        pzt_normals=data["pzt_normals"],
        pzt_tangent_hints=data["pzt_tangent_hints"],
        base_thickness_m=BASE_THICKNESS_M,
        c_hologram_m_s=C_HOLOGRAM_M_S,
        frequency_hz=float(data["frequency_hz"]),
        sound_speed_m_s=float(data["sound_speed_m_s"]),
    )

    print(f"Loaded optimized hologram data from {DATA_PATH}")
    print(f"Saved height render to {HEIGHT_PLOT_PATH}")
    print(f"Saved height arrays to {HEIGHT_DATA_PATH}")
    for output_path, n_facets, nonmanifold_edges in stl_results:
        print(
            f"Saved {output_path.name}: {n_facets} facets, "
            f"{nonmanifold_edges} non-manifold edges"
        )


if __name__ == "__main__":
    main()
