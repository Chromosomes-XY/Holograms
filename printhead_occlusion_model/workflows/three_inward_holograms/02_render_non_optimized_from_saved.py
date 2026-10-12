from pathlib import Path
import importlib.util

THIS_DIR = Path(__file__).resolve().parent
PRINTHEAD_DIR = THIS_DIR.parents[1]
RESULTS_DIR = THIS_DIR / "results"
DATA_PATH = RESULTS_DIR / "non_optimized_spiral_hologram.npz"
OUTPUT_PATH = RESULTS_DIR / "non_optimized_spiral_occlusion_slices.png"


def load_renderer():
    renderer_path = THIS_DIR / "02_render_occlusion_optimized_from_saved.py"
    spec = importlib.util.spec_from_file_location("three_inward_occlusion_renderer", renderer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    renderer = load_renderer()
    renderer.render_saved_hologram(DATA_PATH, OUTPUT_PATH, "non-optimized")


if __name__ == "__main__":
    main()
