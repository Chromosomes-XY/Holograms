from pathlib import Path
import importlib.util
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
REPO_ROOT = PRINTHEAD_DIR.parent
RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "unobstructed_vertical_line_hologram.npz"
STL_DIR = RESULTS_DIR / "hologram_stls"

sys.path.insert(0, str(REPO_ROOT))

from hologramGen import acoustic_params, export_hologram_stl  # noqa: E402


C_HOLOGRAM_M_S = 2550.0
BASE_THICKNESS_M = 0.4e-3
NOTCH_HALF_WIDTH_M = 4.0e-3
NOTCH_OVERLAP_M = 2.0e-3
NOTCH_EXTENSION_M = 4.0e-3
NOTCH_HEIGHT_M = 0.8e-3
DOT_RADIUS_M = 0.65e-3
DOT_SPACING_M = 1.6e-3
DOT_RAISE_M = 0.4e-3


def load_baseline_generator():
    generator_path = THIS_DIR / "01_generate_unobstructed_vertical_line.py"
    spec = importlib.util.spec_from_file_location("free_field_baseline_generator", generator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def make_height_maps(data, apertures, baseline_generator):
    aperture_fields = data["aperture_fields"]
    if len(apertures) != 2 or aperture_fields.shape[0] != 2:
        raise ValueError("The free-field baseline must contain exactly two actuators.")

    frequency_hz = float(data["frequency_hz"])
    sound_speed_m_s = float(data["sound_speed_m_s"])
    height_maps_m = []
    for aperture, aperture_field in zip(apertures, aperture_fields):
        active_mask = baseline_generator.circular_aperture_mask(aperture)
        height_m = phase_to_printable_height_m(
            np.angle(aperture_field),
            frequency_hz=frequency_hz,
            sound_speed_m_s=sound_speed_m_s,
        )
        height_maps_m.append(np.where(active_mask, height_m, np.nan))
    return height_maps_m


def add_top_notch_and_dots(height_map_m, dx_m, diameter_m, dot_count):
    """Add a top identification tab with one or two raised dot markers."""

    if dot_count not in (1, 2):
        raise ValueError("Identification tabs support one or two dots.")

    padding_cells = int(np.ceil((NOTCH_EXTENSION_M + dx_m) / dx_m))
    input_n = height_map_m.shape[0]
    output_n = input_n + 2 * padding_cells
    featured_height_m = np.full((output_n, output_n), np.nan, dtype=float)
    featured_height_m[
        padding_cells:padding_cells + input_n,
        padding_cells:padding_cells + input_n,
    ] = height_map_m

    coords_m = (np.arange(output_n) - output_n / 2.0) * dx_m
    uu_m, vv_m = np.meshgrid(coords_m, coords_m, indexing="xy")
    radius_m = 0.5 * diameter_m
    notch_mask = (
        (np.abs(uu_m) <= NOTCH_HALF_WIDTH_M)
        & (vv_m >= radius_m - NOTCH_OVERLAP_M)
        & (vv_m <= radius_m + NOTCH_EXTENSION_M)
    )
    featured_height_m[notch_mask] = NOTCH_HEIGHT_M

    if dot_count == 1:
        dot_centers_u_m = (0.0,)
    else:
        dot_centers_u_m = (-0.5 * DOT_SPACING_M, 0.5 * DOT_SPACING_M)
    dot_center_v_m = radius_m + 0.5 * NOTCH_EXTENSION_M
    for dot_center_u_m in dot_centers_u_m:
        dot_mask = (
            (uu_m - dot_center_u_m) ** 2
            + (vv_m - dot_center_v_m) ** 2
            <= DOT_RADIUS_M**2
        )
        featured_height_m[dot_mask & notch_mask] = NOTCH_HEIGHT_M + DOT_RAISE_M

    return featured_height_m


def main():
    baseline_generator = load_baseline_generator()
    data = np.load(DATA_PATH)
    apertures = baseline_generator.make_two_usual_actuators()
    height_maps_m = make_height_maps(data, apertures, baseline_generator)

    STL_DIR.mkdir(parents=True, exist_ok=True)
    for idx, (aperture, height_map_m) in enumerate(zip(apertures, height_maps_m)):
        label = chr(ord("a") + idx)
        output_path = STL_DIR / f"pzt_{label}_free_field_hologram.stl"
        featured_height_map_m = add_top_notch_and_dots(
            height_map_m,
            aperture.dx,
            float(data["pzt_diameter_m"]),
            dot_count=idx + 1,
        )
        n_facets, nonmanifold_edges = export_hologram_stl(
            featured_height_map_m,
            aperture.dx,
            output_path=output_path,
        )
        print(
            f"Saved {output_path}: {n_facets} facets, "
            f"{nonmanifold_edges} non-manifold edges"
        )


if __name__ == "__main__":
    main()
