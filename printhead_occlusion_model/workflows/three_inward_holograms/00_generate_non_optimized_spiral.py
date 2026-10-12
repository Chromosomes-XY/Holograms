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
RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "non_optimized_spiral_hologram.npz"
PLOT_PATH = RESULTS_DIR / "non_optimized_spiral_field.png"

sys.path.insert(0, str(REPO_ROOT / "multi_hologram_3d_optimization"))
sys.path.insert(0, str(PRINTHEAD_DIR / "lib"))


def load_optimizer_config():
    optimizer_path = THIS_DIR / "01_optimize_for_occlusion_levels.py"
    spec = importlib.util.spec_from_file_location("three_inward_optimizer_config", optimizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cfg = load_optimizer_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wavelength_m = cfg.SOUND_SPEED_M_S / cfg.FREQUENCY_HZ
    x, y, z, points_m = cfg.make_volume_grid(cfg.VOLUME_BOUNDS_M, n=cfg.VOLUME_N)
    target_extent_m = max(x[-1] - x[0], y[-1] - y[0], z[-1] - z[0])
    demand_segments_m, demand_curve_m = cfg.make_vertical_spiral_segments()
    target_amp = cfg.make_line_segment_target(
        points_m,
        demand_segments_m,
        tube_radius_m=cfg.DEMAND_TUBE_RADIUS_M,
    )
    target_mask = target_amp > 0.0
    if not np.any(target_mask):
        raise ValueError("Demand sampled zero target points. Increase tube radius or grid resolution.")

    apertures = cfg.make_three_actuators()
    initial_aperture_fields = cfg.initialize_from_focus_points(
        apertures,
        points_m[target_mask],
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=cfg.DISTANCE_BIN_M,
    )
    aperture_fields = cfg.apply_circular_apertures(
        apertures,
        [np.exp(1j * np.angle(field)) for field in initial_aperture_fields],
    )

    free_pressure = cfg.forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=target_extent_m,
        distance_bin_m=cfg.DISTANCE_BIN_M,
    )

    np.savez_compressed(
        DATA_PATH,
        x=x,
        y=y,
        z=z,
        target_amp=target_amp,
        free_pressure=free_pressure,
        aperture_fields=np.stack(aperture_fields, axis=0),
        volume_bounds_m=np.asarray(cfg.VOLUME_BOUNDS_M, dtype=float),
        pzt_n_side=cfg.PZT_N_SIDE,
        pzt_width_m=cfg.PZT_WIDTH_M,
        pzt_diameter_m=cfg.PZT_DIAMETER_M,
        pzt_focus_center_m=cfg.PZT_FOCUS_CENTER_M,
        pzt_center_distance_m=cfg.PZT_CENTER_DISTANCE_M,
        pzt_normal_azimuths_deg=np.asarray(cfg.PZT_NORMAL_AZIMUTHS_DEG),
        pzt_count=len(apertures),
        pzt_positions_m=np.asarray([aperture.plane.position_m for aperture in apertures]),
        pzt_normals=np.asarray([aperture.plane.normal for aperture in apertures]),
        pzt_tangent_hints=np.asarray([aperture.plane.tangent_u for aperture in apertures]),
        distance_bin_m=cfg.DISTANCE_BIN_M,
        demand_segments_m=np.array(demand_segments_m, dtype=float),
        demand_curve_m=demand_curve_m,
        demand_spiral_center_m=np.array((cfg.DEMAND_SPIRAL_CENTER_X_M, cfg.DEMAND_SPIRAL_CENTER_Y_M)),
        demand_spiral_radius_m=cfg.DEMAND_SPIRAL_RADIUS_M,
        demand_spiral_turns=cfg.DEMAND_SPIRAL_TURNS,
        demand_spiral_points=cfg.DEMAND_SPIRAL_POINTS,
        demand_tube_radius_m=cfg.DEMAND_TUBE_RADIUS_M,
        frequency_hz=cfg.FREQUENCY_HZ,
        sound_speed_m_s=cfg.SOUND_SPEED_M_S,
        target_extent_m=target_extent_m,
    )

    plot_non_optimized_field(x, y, z, np.abs(free_pressure), target_amp, PLOT_PATH)

    print("Generated non-optimized inward three-hologram spiral")
    print("Target samples:", int(target_mask.sum()))
    print("Mean pressure on spiral:", float(np.abs(free_pressure)[target_mask].mean()))
    print("Mean pressure off spiral:", float(np.abs(free_pressure)[~target_mask].mean()))
    print(f"Saved data to {DATA_PATH}")
    print(f"Saved plot to {PLOT_PATH}")


def plot_non_optimized_field(x, y, z, pressure_amp, target_amp, output_path):
    pressure_grid = pressure_amp.reshape((x.size, y.size, z.size))
    target_grid = target_amp.reshape((x.size, y.size, z.size))
    z_indices = plot_z_indices(z)
    pressure_projection = np.max(pressure_grid[:, :, z_indices], axis=2)
    target_projection = np.max(target_grid[:, :, z_indices], axis=2)

    fig, ax = plt.subplots(figsize=(6.2, 5.3), constrained_layout=True)
    extent = (x[0] * 1e3, x[-1] * 1e3, y[0] * 1e3, y[-1] * 1e3)
    im = ax.imshow(
        pressure_projection.T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=np.percentile(pressure_projection, 99.5),
        aspect="equal",
    )
    if np.any(target_projection > 0.0):
        ax.contour(
            x * 1e3,
            y * 1e3,
            target_projection.T,
            levels=(0.5,),
            colors="cyan",
            linewidths=1.0,
        )
    ax.set_title(f"Non-optimized free field\nz={z[z_indices[0]] * 1e3:.1f}..{z[z_indices[-1]] * 1e3:.1f} mm")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax, shrink=0.82, label="|p|")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_z_indices(z):
    indices = np.flatnonzero((z >= -5e-3) & (z <= 20e-3))
    if indices.size == 0:
        indices = np.arange(z.size)
    return indices


if __name__ == "__main__":
    main()
