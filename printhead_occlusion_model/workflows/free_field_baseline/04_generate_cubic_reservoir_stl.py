"""Generate an open-top cubic reservoir as a watertight ASCII STL."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = (
    THIS_DIR
    / "results"
    / "reservoir_stls"
    / "cubic_reservoir_70mm_outer_2p68mm_wall.stl"
)


def add_quad(vertices, triangles, point_a, point_b, point_c, point_d, normal):
    """Append a consistently oriented quad as two triangles."""

    points = [
        np.asarray(point_a, dtype=float),
        np.asarray(point_b, dtype=float),
        np.asarray(point_c, dtype=float),
        np.asarray(point_d, dtype=float),
    ]
    desired_normal = np.asarray(normal, dtype=float)
    if np.dot(np.cross(points[1] - points[0], points[2] - points[0]), desired_normal) < 0:
        points = [points[0], points[3], points[2], points[1]]

    start = len(vertices)
    vertices.extend(points)
    triangles.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))


def make_open_cubic_reservoir(outer_side_mm=70.0, wall_thickness_mm=2.68):
    """Return vertices and triangles for an open-top cubic reservoir."""

    if outer_side_mm <= 0.0:
        raise ValueError("The outer side length must be positive.")
    if wall_thickness_mm <= 0.0:
        raise ValueError("The wall thickness must be positive.")
    if 2.0 * wall_thickness_mm >= outer_side_mm:
        raise ValueError("The side walls leave no internal cavity.")

    outer_half = 0.5 * outer_side_mm
    inner_half = outer_half - wall_thickness_mm
    inner_bottom = wall_thickness_mm
    top = outer_side_mm

    xi0, xi1 = -inner_half, inner_half
    yi0, yi1 = -inner_half, inner_half
    xo0, xo1 = -outer_half, outer_half
    yo0, yo1 = -outer_half, outer_half
    z0, zi, z1 = 0.0, inner_bottom, top

    vertices = []
    triangles = []

    # Exterior bottom and four exterior walls.
    add_quad(vertices, triangles, (xo0, yo0, z0), (xo1, yo0, z0),
             (xo1, yo1, z0), (xo0, yo1, z0), (0, 0, -1))
    add_quad(vertices, triangles, (xo0, yo0, z0), (xo0, yo1, z0),
             (xo0, yo1, z1), (xo0, yo0, z1), (-1, 0, 0))
    add_quad(vertices, triangles, (xo1, yo0, z0), (xo1, yo0, z1),
             (xo1, yo1, z1), (xo1, yo1, z0), (1, 0, 0))
    add_quad(vertices, triangles, (xo0, yo0, z0), (xo0, yo0, z1),
             (xo1, yo0, z1), (xo1, yo0, z0), (0, -1, 0))
    add_quad(vertices, triangles, (xo0, yo1, z0), (xo1, yo1, z0),
             (xo1, yo1, z1), (xo0, yo1, z1), (0, 1, 0))

    # Interior bottom and four cavity walls; normals point into the cavity.
    add_quad(vertices, triangles, (xi0, yi0, zi), (xi1, yi0, zi),
             (xi1, yi1, zi), (xi0, yi1, zi), (0, 0, 1))
    add_quad(vertices, triangles, (xi0, yi0, zi), (xi0, yi0, z1),
             (xi0, yi1, z1), (xi0, yi1, zi), (1, 0, 0))
    add_quad(vertices, triangles, (xi1, yi0, zi), (xi1, yi1, zi),
             (xi1, yi1, z1), (xi1, yi0, z1), (-1, 0, 0))
    add_quad(vertices, triangles, (xi0, yi0, zi), (xi1, yi0, zi),
             (xi1, yi0, z1), (xi0, yi0, z1), (0, 1, 0))
    add_quad(vertices, triangles, (xi0, yi1, zi), (xi0, yi1, z1),
             (xi1, yi1, z1), (xi1, yi1, zi), (0, -1, 0))

    # Four strips close the wall tops while leaving the reservoir open.
    add_quad(vertices, triangles, (xo0, yo0, z1), (xo1, yo0, z1),
             (xi1, yi0, z1), (xi0, yi0, z1), (0, 0, 1))
    add_quad(vertices, triangles, (xo0, yo1, z1), (xi0, yi1, z1),
             (xi1, yi1, z1), (xo1, yo1, z1), (0, 0, 1))
    add_quad(vertices, triangles, (xo0, yo0, z1), (xi0, yi0, z1),
             (xi0, yi1, z1), (xo0, yo1, z1), (0, 0, 1))
    add_quad(vertices, triangles, (xo1, yo0, z1), (xo1, yo1, z1),
             (xi1, yi1, z1), (xi1, yi0, z1), (0, 0, 1))

    return np.asarray(vertices), np.asarray(triangles, dtype=int)


def triangle_normal(points):
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(normal)
    return normal / length if length else normal


def write_ascii_stl(output_path, vertices, triangles):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("solid cubic_reservoir\n")
        for triangle in triangles:
            points = vertices[triangle]
            normal = triangle_normal(points)
            stream.write(
                "  facet normal {:.9g} {:.9g} {:.9g}\n".format(*normal)
            )
            stream.write("    outer loop\n")
            for point in points:
                stream.write("      vertex {:.9g} {:.9g} {:.9g}\n".format(*point))
            stream.write("    endloop\n  endfacet\n")
        stream.write("endsolid cubic_reservoir\n")


def nonmanifold_edge_count(vertices, triangles):
    edge_counts = {}
    for triangle in triangles:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(vertices[triangle[first]]), tuple(vertices[triangle[second]]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return sum(count != 2 for count in edge_counts.values())


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-side-mm", type=float, default=70.0)
    parser.add_argument("--wall-thickness-mm", type=float, default=2.68)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    vertices, triangles = make_open_cubic_reservoir(
        outer_side_mm=args.outer_side_mm,
        wall_thickness_mm=args.wall_thickness_mm,
    )
    write_ascii_stl(args.output, vertices, triangles)

    dimensions = np.ptp(vertices, axis=0)
    print("Wrote:", args.output)
    print("Outer dimensions [mm]:", dimensions)
    print(
        "Internal cavity [mm]:",
        (
            args.outer_side_mm - 2.0 * args.wall_thickness_mm,
            args.outer_side_mm - 2.0 * args.wall_thickness_mm,
            args.outer_side_mm - args.wall_thickness_mm,
        ),
    )
    print("Wall/base thickness [mm]:", args.wall_thickness_mm)
    print("Triangles:", len(triangles))
    print("Non-manifold edges:", nonmanifold_edge_count(vertices, triangles))


if __name__ == "__main__":
    main()
