from pathlib import Path
import importlib.util

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "non_optimized_spiral_hologram.npz"
HEIGHT_PLOT_PATH = RESULTS_DIR / "non_optimized_hologram_height_arrays.png"
HEIGHT_DATA_PATH = RESULTS_DIR / "non_optimized_hologram_height_arrays.npz"
STL_DIR = RESULTS_DIR / "non_optimized_hologram_stls"


def load_height_exporter():
    exporter_path = THIS_DIR / "04_render_hologram_heights_and_export_stl.py"
    spec = importlib.util.spec_from_file_location("three_inward_height_exporter", exporter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_non_optimized_height_stls(exporter, apertures, height_maps_m):
    STL_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, (aperture, height_map_m) in enumerate(zip(apertures, height_maps_m)):
        label = chr(ord("a") + idx)
        output_path = STL_DIR / f"pzt_{label}_non_optimized_hologram_height.stl"
        n_facets, nonmanifold_edges = exporter.export_hologram_stl(
            height_map_m,
            aperture.dx,
            output_path=output_path,
        )
        results.append((output_path, n_facets, nonmanifold_edges))
    return results


def main():
    exporter = load_height_exporter()
    data = np.load(DATA_PATH)
    apertures = exporter.make_actuators(data)
    height_maps_m, masks = exporter.make_height_maps(data, apertures)

    exporter.plot_height_arrays(data, apertures, height_maps_m, HEIGHT_PLOT_PATH)
    stl_results = export_non_optimized_height_stls(exporter, apertures, height_maps_m)

    np.savez(
        HEIGHT_DATA_PATH,
        height_maps_m=height_maps_m,
        aperture_masks=masks,
        pzt_positions_m=data["pzt_positions_m"],
        pzt_normals=data["pzt_normals"],
        pzt_tangent_hints=data["pzt_tangent_hints"],
        base_thickness_m=exporter.BASE_THICKNESS_M,
        c_hologram_m_s=exporter.C_HOLOGRAM_M_S,
        frequency_hz=float(data["frequency_hz"]),
        sound_speed_m_s=float(data["sound_speed_m_s"]),
    )

    print(f"Loaded non-optimized hologram data from {DATA_PATH}")
    print(f"Saved height render to {HEIGHT_PLOT_PATH}")
    print(f"Saved height arrays to {HEIGHT_DATA_PATH}")
    for output_path, n_facets, nonmanifold_edges in stl_results:
        print(
            f"Saved {output_path.name}: {n_facets} facets, "
            f"{nonmanifold_edges} non-manifold edges"
        )


if __name__ == "__main__":
    main()
