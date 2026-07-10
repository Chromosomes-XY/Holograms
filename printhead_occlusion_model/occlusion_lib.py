from dataclasses import dataclass, field

import numpy as np


def unit_vector(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if norm == 0.0:
        raise ValueError("Vector must be non-zero.")
    return value / norm


@dataclass(frozen=True)
class RadialBottomProfile:
    """Axisymmetric bottom surface sag z(r), measured along the print-head axis."""

    coefficients_m: tuple = (0.0,)

    def sag_m(self, radius_m):
        radius_m = np.asarray(radius_m, dtype=float)
        sag = np.zeros_like(radius_m, dtype=float)
        for power, coefficient_m in enumerate(self.coefficients_m):
            sag += coefficient_m * radius_m**power
        return sag


@dataclass(frozen=True)
class ShellAndBottomPrintHead:
    """Print head with absorbing cylindrical shell and reflective bottom interface."""

    tip_center_m: np.ndarray
    axis: np.ndarray
    radius_m: float
    length_m: float
    shell_thickness_m: float = 1.0e-3
    shell_soft_edge_m: float = 1.0e-3
    shell_transmission: float = 0.05
    bottom_soft_edge_m: float = 0.5e-3
    bottom_transmission: float = 0.04
    bottom_pressure_reflection: complex = -0.98 + 0.0j
    bottom_profile: RadialBottomProfile = field(default_factory=RadialBottomProfile)

    @classmethod
    def from_tip_position(
        cls,
        tip_m,
        axis=(0.0, 0.0, 1.0),
        radius_m=2.5e-3,
        length_m=30e-3,
        shell_thickness_m=1.0e-3,
        shell_soft_edge_m=1.0e-3,
        shell_transmission=0.05,
        bottom_soft_edge_m=0.5e-3,
        bottom_transmission=0.04,
        bottom_pressure_reflection=-0.98 + 0.0j,
        bottom_profile=RadialBottomProfile(),
    ):
        axis = unit_vector(axis)
        return cls(
            tip_center_m=np.asarray(tip_m, dtype=float),
            axis=axis,
            radius_m=radius_m,
            length_m=length_m,
            shell_thickness_m=shell_thickness_m,
            shell_soft_edge_m=shell_soft_edge_m,
            shell_transmission=shell_transmission,
            bottom_soft_edge_m=bottom_soft_edge_m,
            bottom_transmission=bottom_transmission,
            bottom_pressure_reflection=bottom_pressure_reflection,
            bottom_profile=bottom_profile,
        )

    @property
    def shell_start_m(self):
        return self.tip_center_m

    @property
    def shell_end_m(self):
        return self.tip_center_m + self.length_m * self.axis


CylindricalPrintHead = ShellAndBottomPrintHead


def _segment_segment_distance_batch(p0, p1, q0, q1):
    """Distance between many p0-p1 segments and one q0-q1 segment."""

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if p0.ndim == 1:
        p0 = np.broadcast_to(p0, p1.shape)

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0[None, :]

    a = np.einsum("ij,ij->i", u, u)
    b = u @ v
    c = float(np.dot(v, v))
    d = np.einsum("ij,ij->i", u, w)
    e = w @ v
    denom = a * c - b**2

    s = np.zeros(p0.shape[0], dtype=float)
    valid = denom > 1e-18
    s[valid] = np.clip((b[valid] * e[valid] - c * d[valid]) / denom[valid], 0.0, 1.0)

    t = np.clip((b * s + e) / max(c, 1e-18), 0.0, 1.0)

    s = np.clip((b * t - d) / np.maximum(a, 1e-18), 0.0, 1.0)
    t = np.clip((b * s + e) / max(c, 1e-18), 0.0, 1.0)

    closest_p = p0 + s[:, None] * u
    closest_q = q0[None, :] + t[:, None] * v[None, :]
    return np.linalg.norm(closest_p - closest_q, axis=1)


def smoothstep(edge0, edge1, value):
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _ray_cylindrical_wall_intersection(source_m, target_points_m, print_head):
    source_m = np.asarray(source_m, dtype=float)
    target_points_m = np.asarray(target_points_m, dtype=float)
    source_points_m = np.broadcast_to(source_m, target_points_m.shape)
    direction = target_points_m - source_points_m
    rel = source_points_m - print_head.tip_center_m[None, :]

    direction_axial = direction @ print_head.axis
    rel_axial = rel @ print_head.axis
    direction_radial = direction - direction_axial[:, None] * print_head.axis[None, :]
    rel_radial = rel - rel_axial[:, None] * print_head.axis[None, :]

    a = np.einsum("ij,ij->i", direction_radial, direction_radial)
    b = 2.0 * np.einsum("ij,ij->i", rel_radial, direction_radial)
    c = np.einsum("ij,ij->i", rel_radial, rel_radial) - print_head.radius_m**2
    discriminant = b**2 - 4.0 * a * c

    hit = np.zeros(target_points_m.shape[0], dtype=bool)
    valid = (a > 1e-18) & (discriminant >= 0.0)
    if np.any(valid):
        sqrt_disc = np.sqrt(discriminant[valid])
        roots = np.column_stack(
            (
                (-b[valid] - sqrt_disc) / (2.0 * a[valid]),
                (-b[valid] + sqrt_disc) / (2.0 * a[valid]),
            )
        )
        valid_indices = np.flatnonzero(valid)
        for local_col in range(2):
            t = roots[:, local_col]
            on_segment = (t >= 0.0) & (t <= 1.0)
            axial = rel_axial[valid] + t * direction_axial[valid]
            on_shell_length = (axial >= 0.0) & (axial <= print_head.length_m)
            hit[valid_indices] |= on_segment & on_shell_length
    return hit


def ray_shell_transmission_from_source(source_m, target_points_m, print_head):
    """Soft occlusion by the absorbing cylindrical shell."""

    source_m = np.asarray(source_m, dtype=float)
    target_points_m = np.asarray(target_points_m, dtype=float)
    distance_m = _segment_segment_distance_batch(
        source_m,
        target_points_m,
        print_head.shell_start_m,
        print_head.shell_end_m,
    )
    hit_wall = _ray_cylindrical_wall_intersection(source_m, target_points_m, print_head)

    wall_distance_error_m = np.abs(distance_m - print_head.radius_m)
    clear_fraction = smoothstep(
        0.5 * print_head.shell_thickness_m,
        0.5 * print_head.shell_thickness_m + print_head.shell_soft_edge_m,
        wall_distance_error_m,
    )
    shell_shadow = hit_wall.astype(float) * (1.0 - clear_fraction)
    transmission = print_head.shell_transmission + (
        1.0 - print_head.shell_transmission
    ) * (1.0 - shell_shadow)
    return transmission, distance_m, shell_shadow


def _bottom_intersection(source_m, target_points_m, print_head):
    source_m = np.asarray(source_m, dtype=float)
    target_points_m = np.asarray(target_points_m, dtype=float)
    direction = target_points_m - source_m[None, :]
    denom = direction @ print_head.axis
    valid = np.abs(denom) > 1e-15

    t = np.full(target_points_m.shape[0], np.nan)
    t[valid] = ((print_head.tip_center_m - source_m) @ print_head.axis) / denom[valid]
    on_segment = valid & (t >= 0.0) & (t <= 1.0)
    hit_points = source_m[None, :] + np.nan_to_num(t)[:, None] * direction
    radial_vector = hit_points - print_head.tip_center_m[None, :]
    axial_offset = radial_vector @ print_head.axis
    radial_vector = radial_vector - axial_offset[:, None] * print_head.axis[None, :]
    radius_m = np.linalg.norm(radial_vector, axis=1)

    # Initial profile is flat. Nonzero radial sag can later move the effective
    # interface position along the axis; this remains a paraxial approximation.
    sag_m = print_head.bottom_profile.sag_m(radius_m)
    profile_error_m = np.abs(axial_offset - sag_m)
    near_profile = profile_error_m <= max(print_head.bottom_soft_edge_m, 1e-12)
    return on_segment & near_profile, radius_m


def ray_bottom_transmission_from_source(source_m, target_points_m, print_head):
    """Transmission through the flat/axisymmetric fluid-air bottom interface."""

    intersects, radius_m = _bottom_intersection(source_m, target_points_m, print_head)
    inside_fraction = 1.0 - smoothstep(
        print_head.radius_m,
        print_head.radius_m + print_head.bottom_soft_edge_m,
        radius_m,
    )
    bottom_shadow = intersects.astype(float) * inside_fraction
    transmission = 1.0 - bottom_shadow * (1.0 - print_head.bottom_transmission)
    return transmission, bottom_shadow


def ray_transmission_from_source(source_m, target_points_m, print_head):
    shell_transmission, shell_distance_m, shell_shadow = ray_shell_transmission_from_source(
        source_m,
        target_points_m,
        print_head,
    )
    bottom_transmission, bottom_shadow = ray_bottom_transmission_from_source(
        source_m,
        target_points_m,
        print_head,
    )
    return shell_transmission * bottom_transmission, {
        "shell_distance_m": shell_distance_m,
        "shell_shadow": shell_shadow,
        "bottom_shadow": bottom_shadow,
    }


def aperture_visibility(apertures, points_m, print_head):
    visibility = np.ones((len(apertures), points_m.shape[0]), dtype=float)
    diagnostics = []
    for idx, aperture in enumerate(apertures):
        transmission, aperture_diagnostics = ray_transmission_from_source(
            aperture.plane.position_m,
            points_m,
            print_head,
        )
        visibility[idx] = transmission
        diagnostics.append(aperture_diagnostics)
    return visibility, diagnostics


def aperture_element_positions(aperture, stride=4):
    """Sample physical source positions across an aperture face."""

    coords_m = aperture.coords_m[::stride]
    uu, vv = np.meshgrid(coords_m, coords_m, indexing="xy")
    return (
        aperture.plane.position_m[None, :]
        + uu.ravel()[:, None] * aperture.plane.tangent_u[None, :]
        + vv.ravel()[:, None] * aperture.plane.tangent_v[None, :]
    )


def sampled_aperture_visibility(aperture, points_m, print_head, stride=4):
    """Average ray visibility over sampled source elements on one aperture."""

    sources_m = aperture_element_positions(aperture, stride=stride)
    visibility = np.zeros(points_m.shape[0], dtype=float)
    shell_shadow = np.zeros(points_m.shape[0], dtype=float)
    bottom_shadow = np.zeros(points_m.shape[0], dtype=float)

    for source_m in sources_m:
        transmission, diagnostics = ray_transmission_from_source(
            source_m,
            points_m,
            print_head,
        )
        visibility += transmission
        shell_shadow += diagnostics["shell_shadow"]
        bottom_shadow += diagnostics["bottom_shadow"]

    scale = 1.0 / sources_m.shape[0]
    diagnostics = {
        "shell_shadow": shell_shadow * scale,
        "bottom_shadow": bottom_shadow * scale,
        "source_count": sources_m.shape[0],
    }
    return visibility * scale, diagnostics


def apply_partial_occlusion_by_aperture(contributions, apertures, points_m, print_head):
    visibility, _ = aperture_visibility(apertures, points_m, print_head)
    total = np.zeros(points_m.shape[0], dtype=np.complex128)
    for idx, contribution in enumerate(contributions):
        total += visibility[idx] * contribution
    return total, visibility
