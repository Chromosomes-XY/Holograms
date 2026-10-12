"""Export printable binary STLs for the current point and Q=1 holograms."""

from pathlib import Path
import struct

import numpy as np


RESULT_DIR = Path("intensity_forced_point_and_q1_f25mm_2164khz")
RESULT_FILE = RESULT_DIR / "intensity_forced_point_and_q1_f25mm_2164khz.npz"
BASE_THICKNESS_MM = 0.30
SAFE_MIN_TOTAL_MM = 0.40
SAFE_MAX_TOTAL_MM = 2.00
SAFE_EDGE_TAPER_MM = 0.50


def facet_record(a, b, c):
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
    values = (*normal, *a, *b, *c, 0)
    return struct.pack("<12fH", *values)


def export_binary_stl(path, relief_m, dx_m, base_thickness_mm, name):
    relief_mm = np.asarray(relief_m, dtype=float) * 1e3
    valid_vertices = np.isfinite(relief_mm)
    top_height = np.where(valid_vertices, relief_mm + base_thickness_mm, 0.0)
    n = relief_mm.shape[0]
    coords_mm = (np.arange(n) - n / 2) * dx_m * 1e3

    valid_cells = (
        valid_vertices[:-1, :-1]
        & valid_vertices[1:, :-1]
        & valid_vertices[:-1, 1:]
        & valid_vertices[1:, 1:]
    )
    cell_count = int(np.count_nonzero(valid_cells))
    boundary_count = 0
    indices = np.argwhere(valid_cells)
    for i, j in indices:
        boundary_count += i == 0 or not valid_cells[i - 1, j]
        boundary_count += i == n - 2 or not valid_cells[i + 1, j]
        boundary_count += j == 0 or not valid_cells[i, j - 1]
        boundary_count += j == n - 2 or not valid_cells[i, j + 1]
    triangle_count = 4 * cell_count + 2 * int(boundary_count)

    header = f"{name}; base={base_thickness_mm:.3f}mm".encode("ascii")[:80]
    header = header + b" " * (80 - len(header))
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", triangle_count))

        for i, j in indices:
            x0, x1 = coords_mm[j], coords_mm[j + 1]
            y0, y1 = coords_mm[i], coords_mm[i + 1]
            tl = np.array((x0, y0, top_height[i, j]), dtype=np.float32)
            tr = np.array((x1, y0, top_height[i, j + 1]), dtype=np.float32)
            bl = np.array((x0, y1, top_height[i + 1, j]), dtype=np.float32)
            br = np.array((x1, y1, top_height[i + 1, j + 1]), dtype=np.float32)
            btl = np.array((x0, y0, 0.0), dtype=np.float32)
            btr = np.array((x1, y0, 0.0), dtype=np.float32)
            bbl = np.array((x0, y1, 0.0), dtype=np.float32)
            bbr = np.array((x1, y1, 0.0), dtype=np.float32)

            handle.write(facet_record(tl, tr, bl))
            handle.write(facet_record(tr, br, bl))
            handle.write(facet_record(btl, bbl, btr))
            handle.write(facet_record(btr, bbl, bbr))

            if i == 0 or not valid_cells[i - 1, j]:
                handle.write(facet_record(tl, btl, tr))
                handle.write(facet_record(tr, btl, btr))
            if i == n - 2 or not valid_cells[i + 1, j]:
                handle.write(facet_record(bl, br, bbl))
                handle.write(facet_record(br, bbr, bbl))
            if j == 0 or not valid_cells[i, j - 1]:
                handle.write(facet_record(tl, bl, btl))
                handle.write(facet_record(bl, bbl, btl))
            if j == n - 2 or not valid_cells[i, j + 1]:
                handle.write(facet_record(tr, btr, br))
                handle.write(facet_record(br, btr, bbr))

    expected_size = 84 + 50 * triangle_count
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"STL size mismatch for {path}: expected {expected_size}, got {actual_size}."
        )
    return triangle_count, actual_size


def make_safe_ring_relief(relief_m, dx_m):
    """Clamp the phase relief and fade it onto a uniform perimeter floor."""
    relief_mm = np.asarray(relief_m, dtype=float) * 1e3
    valid = np.isfinite(relief_mm)
    n = relief_mm.shape[0]
    coords_mm = (np.arange(n) - n / 2) * dx_m * 1e3
    xv, yv = np.meshgrid(coords_mm, coords_mm, indexing="xy")
    radius_mm = np.hypot(xv, yv)
    aperture_radius_mm = 10.0

    maximum_relief_mm = SAFE_MAX_TOTAL_MM - SAFE_MIN_TOTAL_MM
    safe_relief_mm = np.clip(relief_mm, 0.0, maximum_relief_mm)
    taper_coordinate = np.clip(
        (aperture_radius_mm - radius_mm) / SAFE_EDGE_TAPER_MM,
        0.0,
        1.0,
    )
    smooth_taper = taper_coordinate**2 * (3.0 - 2.0 * taper_coordinate)
    safe_relief_mm *= smooth_taper
    safe_relief_mm[~valid] = np.nan
    return safe_relief_mm * 1e-3, smooth_taper


def main():
    data = np.load(RESULT_FILE)
    dx_m = float(data["dx_m"])
    exports = (
        (
            "point_thickness_m",
            RESULT_DIR / "point_focus_f25mm_2164khz_20mm_pzt.stl",
            "intensity-optimized point focus",
        ),
        (
            "ring_thickness_m",
            RESULT_DIR / "q1_ring_f25mm_2164khz_20mm_pzt.stl",
            "analytic Q1 ring focus",
        ),
    )
    for key, path, name in exports:
        if path.exists():
            print(f"Keeping existing validated STL: {path}")
            continue
        triangles, size = export_binary_stl(
            path,
            data[key],
            dx_m,
            BASE_THICKNESS_MM,
            name,
        )
        total = np.asarray(data[key]) * 1e3 + BASE_THICKNESS_MM
        print(
            f"{path}: {triangles:,} triangles, {size/1e6:.2f} MB, "
            f"total thickness {np.nanmin(total):.4f}–{np.nanmax(total):.4f} mm"
        )

    safe_relief_m, edge_taper = make_safe_ring_relief(
        data["ring_thickness_m"],
        dx_m,
    )
    # A 0.094 mm STL mesh is sufficient for the printer-scale safety rim and
    # avoids duplicating the much larger simulation grid in the geometry file.
    safe_relief_export_m = safe_relief_m[::2, ::2]
    safe_export_dx_m = 2.0 * dx_m
    safe_path = RESULT_DIR / "q1_ring_f25mm_2164khz_20mm_pzt_safe_edge.stl"
    triangles, size = export_binary_stl(
        safe_path,
        safe_relief_export_m,
        safe_export_dx_m,
        SAFE_MIN_TOTAL_MM,
        "analytic Q1 ring focus; safe tapered edge",
    )
    safe_total_mm = safe_relief_m * 1e3 + SAFE_MIN_TOTAL_MM
    np.savez_compressed(
        RESULT_DIR / "q1_ring_safe_edge_geometry.npz",
        safe_relief_m=safe_relief_m,
        safe_total_thickness_m=safe_total_mm * 1e-3,
        edge_taper=edge_taper,
        minimum_total_thickness_mm=SAFE_MIN_TOTAL_MM,
        maximum_total_thickness_mm=SAFE_MAX_TOTAL_MM,
        edge_taper_width_mm=SAFE_EDGE_TAPER_MM,
    )
    print(
        f"{safe_path}: {triangles:,} triangles, {size/1e6:.2f} MB, "
        f"safe total thickness {np.nanmin(safe_total_mm):.4f}–"
        f"{np.nanmax(safe_total_mm):.4f} mm; "
        f"uniform {SAFE_EDGE_TAPER_MM:.2f} mm perimeter taper"
    )


if __name__ == "__main__":
    main()
