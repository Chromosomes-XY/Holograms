from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftfreq, ifft2


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OrientedPlane:
    name: str
    position_m: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray

    @classmethod
    def from_position_and_normal(
        cls,
        name,
        position_m,
        normal,
        tangent_hint=(0.0, 0.0, 1.0),
    ):
        position_m = np.asarray(position_m, dtype=float)
        normal = unit_vector(normal)
        tangent_hint = np.asarray(tangent_hint, dtype=float)
        tangent_u = tangent_hint - np.dot(tangent_hint, normal) * normal
        if np.linalg.norm(tangent_u) < 1e-12:
            tangent_hint = np.array((0.0, 1.0, 0.0))
            tangent_u = tangent_hint - np.dot(tangent_hint, normal) * normal
        tangent_u = unit_vector(tangent_u)
        tangent_v = unit_vector(np.cross(normal, tangent_u))
        return cls(name, position_m, normal, tangent_u, tangent_v)


@dataclass
class HologramAperture:
    plane: OrientedPlane
    n_side: int
    width_m: float
    coords_m: np.ndarray
    uu_mm: np.ndarray
    vv_mm: np.ndarray
    transfer_cache: dict

    @property
    def dx(self):
        return self.width_m / self.n_side

    @property
    def shape(self):
        return (self.n_side, self.n_side)

    @property
    def size(self):
        return self.n_side * self.n_side


@dataclass(frozen=True)
class PlanarWallModel:
    """Parallel fluid-wall-fluid layers measured from an aperture plane."""

    frequency_hz: float
    outer_sound_speed_m_s: float
    outer_density_kg_m3: float
    wall_offset_m: float
    wall_thickness_m: float
    wall_sound_speed_m_s: float
    wall_density_kg_m3: float
    inner_sound_speed_m_s: float
    inner_density_kg_m3: float

    def __post_init__(self):
        positive_values = (
            self.frequency_hz,
            self.outer_sound_speed_m_s,
            self.outer_density_kg_m3,
            self.wall_thickness_m,
            self.wall_sound_speed_m_s,
            self.wall_density_kg_m3,
            self.inner_sound_speed_m_s,
            self.inner_density_kg_m3,
        )
        if any(value <= 0.0 for value in positive_values):
            raise ValueError("Wall and medium properties must be positive.")
        if self.wall_offset_m < 0.0:
            raise ValueError("Wall offset must be non-negative.")


def unit_vector(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("Vector must be non-zero.")
    return value / norm


def make_hologram_aperture(plane, n_side, width_m):
    dx = width_m / n_side
    coords_m = (np.arange(n_side) - (n_side - 1) / 2.0) * dx
    uu, vv = np.meshgrid(coords_m, coords_m, indexing="xy")
    return HologramAperture(
        plane=plane,
        n_side=n_side,
        width_m=width_m,
        coords_m=coords_m,
        uu_mm=uu * 1e3,
        vv_mm=vv * 1e3,
        transfer_cache={},
    )


def make_volume_grid(bounds_m, n=(34, 34, 24)):
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_m
    x = np.linspace(xmin, xmax, n[0])
    y = np.linspace(ymin, ymax, n[1])
    z = np.linspace(zmin, zmax, n[2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))


def make_line_segment_target(points_m, segments_m, tube_radius_m):
    target = np.zeros(points_m.shape[0], dtype=float)
    for start_m, end_m in segments_m:
        start_m = np.asarray(start_m, dtype=float)
        end_m = np.asarray(end_m, dtype=float)
        segment = end_m - start_m
        length_sq = float(np.dot(segment, segment))
        if length_sq == 0.0:
            distance = np.linalg.norm(points_m - start_m, axis=1)
        else:
            t = np.clip(((points_m - start_m) @ segment) / length_sq, 0.0, 1.0)
            closest = start_m[None, :] + t[:, None] * segment[None, :]
            distance = np.linalg.norm(points_m - closest, axis=1)
        target = np.maximum(target, (distance <= tube_radius_m).astype(float))
    return target


def make_optimization_samples(points_m, target_amp, max_background, seed=11):
    rng = np.random.default_rng(seed)
    target_idx = np.flatnonzero(target_amp > 0.0)
    background_idx = np.flatnonzero(target_amp <= 0.0)
    if background_idx.size > max_background:
        background_idx = rng.choice(background_idx, size=max_background, replace=False)
    sample_idx = np.concatenate((target_idx, background_idx))
    desired_amp = np.zeros(sample_idx.size, dtype=float)
    weights = np.full(sample_idx.size, 0.05, dtype=float)
    desired_amp[: target_idx.size] = 1.0
    weights[: target_idx.size] = 80.0
    return sample_idx, points_m[sample_idx], desired_amp, weights, target_idx.size


def point_local_coordinates(points_m, plane):
    rel = points_m - plane.position_m[None, :]
    u = rel @ plane.tangent_u
    v = rel @ plane.tangent_v
    s = rel @ plane.normal
    return u, v, s


def angular_spectrum_transfer(aperture, wavelength_m, distance_m, target_extent_m):
    key = (
        round(float(distance_m), 12),
        round(float(wavelength_m), 12),
        round(float(target_extent_m), 12),
    )
    if key in aperture.transfer_cache:
        return aperture.transfer_cache[key]

    n = aperture.n_side
    k = 2.0 * np.pi / wavelength_m
    ku = 2.0 * np.pi * fftfreq(n, d=aperture.dx)
    kv = 2.0 * np.pi * fftfreq(n, d=aperture.dx)
    kuv, kvv = np.meshgrid(ku, kv, indexing="xy")
    k_perp2 = kuv**2 + kvv**2

    # Finite-aperture / finite-target cutoff from the paper's modified propagator.
    target_half_depth_m = max(abs(distance_m), aperture.dx)
    cutoff = k / np.sqrt(
        1.0 + 4.0 * target_half_depth_m**2 / (target_extent_m + aperture.width_m) ** 2
    )
    valid = k_perp2 <= cutoff**2
    kz = np.sqrt(np.maximum(cutoff**2 - k_perp2, 0.0))
    transfer = np.zeros((n, n), dtype=np.complex128)
    transfer[valid] = np.exp(1j * distance_m * kz[valid])
    aperture.transfer_cache[key] = transfer
    return transfer


def _axial_wavenumber(wavenumber, transverse_wavenumber_sq):
    return np.sqrt((wavenumber + 0.0j) ** 2 - transverse_wavenumber_sq)


def _normal_impedance(density_kg_m3, omega_rad_s, axial_wavenumber):
    safe_kz = np.where(
        np.abs(axial_wavenumber) > 1e-12,
        axial_wavenumber,
        1e-12 + 0.0j,
    )
    return density_kg_m3 * omega_rad_s / safe_kz


def layered_angular_spectrum_transfer(
    aperture,
    distance_m,
    target_extent_m,
    wall_model,
):
    """Transfer through an outer fluid, acrylic wall, and inner fluid.

    Transverse wavenumber is conserved at both parallel interfaces. Pressure
    transmission includes angle-dependent normal impedance and the coherent
    multiple reflections inside the wall slab.
    """

    key = (
        "planar_wall",
        round(float(distance_m), 12),
        round(float(target_extent_m), 12),
        wall_model,
    )
    if key in aperture.transfer_cache:
        return aperture.transfer_cache[key]

    n = aperture.n_side
    omega = 2.0 * np.pi * wall_model.frequency_hz
    k_outer = omega / wall_model.outer_sound_speed_m_s
    k_wall = omega / wall_model.wall_sound_speed_m_s
    k_inner = omega / wall_model.inner_sound_speed_m_s
    ku = 2.0 * np.pi * fftfreq(n, d=aperture.dx)
    kv = 2.0 * np.pi * fftfreq(n, d=aperture.dx)
    kuv, kvv = np.meshgrid(ku, kv, indexing="xy")
    k_perp2 = kuv**2 + kvv**2

    target_half_depth_m = max(abs(distance_m), aperture.dx)
    cutoff = k_outer / np.sqrt(
        1.0 + 4.0 * target_half_depth_m**2
        / (target_extent_m + aperture.width_m) ** 2
    )
    valid = k_perp2 <= cutoff**2
    kz_outer = _axial_wavenumber(k_outer, k_perp2)
    kz_wall = _axial_wavenumber(k_wall, k_perp2)
    kz_inner = _axial_wavenumber(k_inner, k_perp2)

    z_outer = _normal_impedance(
        wall_model.outer_density_kg_m3,
        omega,
        kz_outer,
    )
    z_wall = _normal_impedance(
        wall_model.wall_density_kg_m3,
        omega,
        kz_wall,
    )
    z_inner = _normal_impedance(
        wall_model.inner_density_kg_m3,
        omega,
        kz_inner,
    )
    t_outer_wall = 2.0 * z_wall / (z_outer + z_wall)
    t_wall_inner = 2.0 * z_inner / (z_wall + z_inner)
    r_wall_outer = (z_outer - z_wall) / (z_outer + z_wall)
    r_wall_inner = (z_inner - z_wall) / (z_inner + z_wall)
    wall_phase = np.exp(1j * kz_wall * wall_model.wall_thickness_m)
    slab_denominator = 1.0 - r_wall_outer * r_wall_inner * wall_phase**2
    forward_wall_amplitude = t_outer_wall / slab_denominator
    effective_reflection = (
        forward_wall_amplitude * (1.0 + r_wall_inner * wall_phase**2) - 1.0
    )

    transfer = np.zeros((n, n), dtype=np.complex128)
    if distance_m <= wall_model.wall_offset_m:
        incident = np.exp(1j * kz_outer * distance_m)
        reflected = effective_reflection * np.exp(
            1j * kz_outer * (2.0 * wall_model.wall_offset_m - distance_m)
        )
        transfer[valid] = (incident + reflected)[valid]
    elif distance_m <= wall_model.wall_offset_m + wall_model.wall_thickness_m:
        distance_in_wall_m = distance_m - wall_model.wall_offset_m
        field_in_wall = forward_wall_amplitude * (
            np.exp(1j * kz_wall * distance_in_wall_m)
            + r_wall_inner
            * np.exp(
                1j
                * kz_wall
                * (2.0 * wall_model.wall_thickness_m - distance_in_wall_m)
            )
        )
        transfer[valid] = (
            np.exp(1j * kz_outer[valid] * wall_model.wall_offset_m)
            * field_in_wall[valid]
        )
    else:
        slab_transmission = (
            forward_wall_amplitude * t_wall_inner * wall_phase
        )
        distance_in_inner_m = (
            distance_m
            - wall_model.wall_offset_m
            - wall_model.wall_thickness_m
        )
        transfer[valid] = (
            np.exp(1j * kz_outer[valid] * wall_model.wall_offset_m)
            * slab_transmission[valid]
            * np.exp(1j * kz_inner[valid] * distance_in_inner_m)
        )

    aperture.transfer_cache[key] = transfer
    return transfer


def propagate_aperture(
    aperture_field,
    aperture,
    wavelength_m,
    distance_m,
    target_extent_m,
    propagation_model=None,
):
    if propagation_model is None:
        transfer = angular_spectrum_transfer(
            aperture,
            wavelength_m=wavelength_m,
            distance_m=distance_m,
            target_extent_m=target_extent_m,
        )
    else:
        transfer = layered_angular_spectrum_transfer(
            aperture,
            distance_m=distance_m,
            target_extent_m=target_extent_m,
            wall_model=propagation_model,
        )
    return ifft2(fft2(aperture_field) * transfer)


def adjoint_propagate(
    field_grad,
    aperture,
    wavelength_m,
    distance_m,
    target_extent_m,
    propagation_model=None,
):
    if propagation_model is None:
        transfer = angular_spectrum_transfer(
            aperture,
            wavelength_m=wavelength_m,
            distance_m=distance_m,
            target_extent_m=target_extent_m,
        )
    else:
        transfer = layered_angular_spectrum_transfer(
            aperture,
            distance_m=distance_m,
            target_extent_m=target_extent_m,
            wall_model=propagation_model,
        )
    return ifft2(fft2(field_grad) * np.conj(transfer))


def bilinear_sample(field, coords_m, u, v):
    dx = coords_m[1] - coords_m[0]
    raw_u_idx = (u - coords_m[0]) / dx
    raw_v_idx = (v - coords_m[0]) / dx
    valid = (
        (raw_v_idx >= 0.0)
        & (raw_u_idx >= 0.0)
        & (raw_v_idx <= field.shape[0] - 1)
        & (raw_u_idx <= field.shape[1] - 1)
    )

    # Keep exact edge samples usable while retaining the existing 4-point stencil.
    eps = 1e-9
    u_idx = np.clip(raw_u_idx, 0.0, field.shape[1] - 1 - eps)
    v_idx = np.clip(raw_v_idx, 0.0, field.shape[0] - 1 - eps)
    j0 = np.floor(u_idx).astype(int)
    i0 = np.floor(v_idx).astype(int)
    du = u_idx - j0
    dv = v_idx - i0
    values = np.zeros(u.shape, dtype=np.complex128)
    if np.any(valid):
        ii = i0[valid]
        jj = j0[valid]
        duv = du[valid]
        dvv = dv[valid]
        values[valid] = (
            (1.0 - duv) * (1.0 - dvv) * field[ii, jj]
            + duv * (1.0 - dvv) * field[ii, jj + 1]
            + (1.0 - duv) * dvv * field[ii + 1, jj]
            + duv * dvv * field[ii + 1, jj + 1]
        )
    return values, valid, i0, j0, du, dv


def bilinear_adjoint(values_grad, valid, i0, j0, du, dv, shape):
    grad = np.zeros(shape, dtype=np.complex128)
    if np.any(valid):
        ii = i0[valid]
        jj = j0[valid]
        duv = du[valid]
        dvv = dv[valid]
        g = values_grad[valid]
        np.add.at(grad, (ii, jj), (1.0 - duv) * (1.0 - dvv) * g)
        np.add.at(grad, (ii, jj + 1), duv * (1.0 - dvv) * g)
        np.add.at(grad, (ii + 1, jj), (1.0 - duv) * dvv * g)
        np.add.at(grad, (ii + 1, jj + 1), duv * dvv * g)
    return grad


def distance_groups(s, forward, distance_bin_m=None):
    if distance_bin_m is None:
        binned_s = np.round(s, 12)
        distances = np.unique(binned_s[forward])
        return [
            (float(distance), forward & np.isclose(binned_s, distance, atol=1e-12))
            for distance in distances
        ]

    binned_s = np.round(s / distance_bin_m) * distance_bin_m
    distances = np.unique(binned_s[forward])
    return [(float(distance), forward & (binned_s == distance)) for distance in distances]


def forward_pressure(
    points_m,
    apertures,
    aperture_fields,
    wavelength_m,
    target_extent_m,
    distance_bin_m=None,
    propagation_models=None,
):
    if propagation_models is None:
        propagation_models = [None] * len(apertures)
    if len(propagation_models) != len(apertures):
        raise ValueError("Provide one propagation model per aperture.")
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    for aperture, aperture_field, propagation_model in zip(
        apertures,
        aperture_fields,
        propagation_models,
    ):
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        forward = s > 0
        for distance, slice_mask in distance_groups(s, forward, distance_bin_m):
            propagated = propagate_aperture(
                aperture_field,
                aperture,
                wavelength_m,
                float(distance),
                target_extent_m,
                propagation_model=propagation_model,
            )
            values, *_ = bilinear_sample(
                propagated,
                aperture.coords_m,
                u[slice_mask],
                v[slice_mask],
            )
            pressure[slice_mask] += values
    return pressure


def amplitude_loss_and_gradient(
    points_m,
    apertures,
    aperture_fields,
    pressure,
    desired_amp,
    loss_weights,
    pressure_scale,
    wavelength_m,
    target_extent_m,
    distance_bin_m=None,
    propagation_models=None,
):
    if propagation_models is None:
        propagation_models = [None] * len(apertures)
    if len(propagation_models) != len(apertures):
        raise ValueError("Provide one propagation model per aperture.")
    eps = 1e-12
    amp = np.abs(pressure)
    residual = amp / pressure_scale - desired_amp
    normalizer = np.sum(loss_weights)
    loss = float(np.sum(loss_weights * residual**2) / normalizer)
    grad_pressure = (
        (2.0 * loss_weights / normalizer)
        * residual
        * pressure
        / ((amp + eps) * pressure_scale)
    )

    aperture_grads = []
    for aperture, propagation_model in zip(apertures, propagation_models):
        grad_aperture = np.zeros(aperture.shape, dtype=np.complex128)
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        forward = s > 0
        for distance, slice_mask in distance_groups(s, forward, distance_bin_m):
            _, valid, i0, j0, du, dv = bilinear_sample(
                np.zeros(aperture.shape, dtype=np.complex128),
                aperture.coords_m,
                u[slice_mask],
                v[slice_mask],
            )
            slice_grad = bilinear_adjoint(
                grad_pressure[slice_mask],
                valid,
                i0,
                j0,
                du,
                dv,
                aperture.shape,
            )
            grad_aperture += adjoint_propagate(
                slice_grad,
                aperture,
                wavelength_m,
                float(distance),
                target_extent_m,
                propagation_model=propagation_model,
            )
        aperture_grads.append(grad_aperture)

    return loss, amp, aperture_grads


def project_unit_phase(aperture_fields):
    return [np.exp(1j * np.angle(field)) for field in aperture_fields]


def initialize_from_focus_points(
    apertures,
    focus_points_m,
    wavelength_m,
    target_extent_m,
    distance_bin_m=None,
    propagation_models=None,
):
    if propagation_models is None:
        propagation_models = [None] * len(apertures)
    if len(propagation_models) != len(apertures):
        raise ValueError("Provide one propagation model per aperture.")
    aperture_fields = []
    for aperture, propagation_model in zip(apertures, propagation_models):
        seed_field = np.zeros(aperture.shape, dtype=np.complex128)
        u, v, s = point_local_coordinates(np.asarray(focus_points_m), aperture.plane)
        for distance, point_mask in distance_groups(s, s > 0, distance_bin_m):
            _, valid, i0, j0, du, dv = bilinear_sample(
                np.zeros(aperture.shape, dtype=np.complex128),
                aperture.coords_m,
                u[point_mask],
                v[point_mask],
            )
            focus_values = np.ones(np.count_nonzero(point_mask), dtype=np.complex128)
            focus_slice = bilinear_adjoint(
                focus_values,
                valid,
                i0,
                j0,
                du,
                dv,
                aperture.shape,
            )
            seed_field += adjoint_propagate(
                focus_slice,
                aperture,
                wavelength_m,
                float(distance),
                target_extent_m,
                propagation_model=propagation_model,
            )
        if np.max(np.abs(seed_field)) < 1e-12:
            seed_field = np.ones(aperture.shape, dtype=np.complex128)
        aperture_fields.append(np.exp(1j * np.angle(seed_field)))
    return aperture_fields


def aperture_visibility_report(apertures, points_m):
    reports = []
    if points_m.size == 0:
        return [
            {
                "name": aperture.plane.name,
                "front_count": 0,
                "aperture_count": 0,
                "usable_count": 0,
                "total_count": 0,
            }
            for aperture in apertures
        ]

    for aperture in apertures:
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        lower = aperture.coords_m[0]
        upper = aperture.coords_m[-1]
        in_front = s > 0.0
        in_aperture = (u >= lower) & (u <= upper) & (v >= lower) & (v <= upper)
        reports.append(
            {
                "name": aperture.plane.name,
                "front_count": int(np.count_nonzero(in_front)),
                "aperture_count": int(np.count_nonzero(in_aperture)),
                "usable_count": int(np.count_nonzero(in_front & in_aperture)),
                "total_count": int(points_m.shape[0]),
            }
        )
    return reports


def flatten_fields(fields):
    return np.concatenate([field.ravel() for field in fields])


def evaluate_amplitude_loss(
    points_m,
    apertures,
    aperture_fields,
    desired_amp,
    loss_weights,
    pressure_scale,
    wavelength_m,
    target_extent_m,
    distance_bin_m=None,
    propagation_models=None,
):
    pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m,
        distance_bin_m=distance_bin_m,
        propagation_models=propagation_models,
    )
    amp = np.abs(pressure)
    residual = amp / pressure_scale - desired_amp
    loss = float(np.sum(loss_weights * residual**2) / np.sum(loss_weights))
    return loss, amp, pressure


def optimize_projected_phase(
    points_m,
    apertures,
    desired_amp,
    loss_weights,
    pressure_scale,
    target_count,
    wavelength_m,
    target_extent_m,
    n_iter=120,
    initial_step=2.0,
    max_step=32.0,
    beta=0.5,
    sigma=0.1,
    seed=5,
    initial_aperture_fields=None,
    distance_bin_m=None,
    propagation_models=None,
):
    if initial_aperture_fields is None:
        rng = np.random.default_rng(seed)
        aperture_fields = [
            np.exp(1j * rng.uniform(-np.pi, np.pi, aperture.shape))
            for aperture in apertures
        ]
    else:
        aperture_fields = project_unit_phase(initial_aperture_fields)
    step = initial_step
    history = {"loss": [], "target_mean": [], "background_mean": []}

    for iteration in range(1, n_iter + 1):
        pressure = forward_pressure(
            points_m,
            apertures,
            aperture_fields,
            wavelength_m,
            target_extent_m,
            distance_bin_m=distance_bin_m,
            propagation_models=propagation_models,
        )
        loss, amp, aperture_grads = amplitude_loss_and_gradient(
            points_m,
            apertures,
            aperture_fields,
            pressure,
            desired_amp,
            loss_weights,
            pressure_scale,
            wavelength_m,
            target_extent_m,
            distance_bin_m=distance_bin_m,
            propagation_models=propagation_models,
        )
        grad_norm_sq = float(
            sum(np.vdot(grad, grad).real for grad in aperture_grads)
        )

        local_step = step
        while True:
            candidate = project_unit_phase(
                [
                    field - local_step * grad
                    for field, grad in zip(aperture_fields, aperture_grads)
                ]
            )
            candidate_loss, _, _ = evaluate_amplitude_loss(
                points_m,
                apertures,
                candidate,
                desired_amp,
                loss_weights,
                pressure_scale,
                wavelength_m,
                target_extent_m,
                distance_bin_m=distance_bin_m,
                propagation_models=propagation_models,
            )
            if candidate_loss <= loss - sigma * local_step * grad_norm_sq / 2.0:
                break
            local_step *= beta
            if local_step < 1e-8:
                break

        aperture_fields = candidate
        step = min(local_step / beta, max_step)
        history["loss"].append(loss)
        history["target_mean"].append(float(amp[:target_count].mean()))
        history["background_mean"].append(float(amp[target_count:].mean()))

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"iter {iteration:03d} loss={loss:.6e} "
                f"target_mean={history['target_mean'][-1]:.6e} "
                f"background_mean={history['background_mean'][-1]:.6e} "
                f"step={local_step:.2e}"
            )

    return aperture_fields, history


def optimize_phase_adam(
    points_m,
    apertures,
    desired_amp,
    loss_weights,
    pressure_scale,
    target_count,
    wavelength_m,
    target_extent_m,
    n_iter=160,
    learning_rate=0.08,
    seed=5,
    initial_aperture_fields=None,
    distance_bin_m=None,
    aperture_masks=None,
    propagation_models=None,
):
    if aperture_masks is None:
        aperture_masks = [np.ones(aperture.shape, dtype=bool) for aperture in apertures]
    if len(aperture_masks) != len(apertures):
        raise ValueError("Provide one aperture mask per aperture.")
    aperture_masks = [np.asarray(mask, dtype=bool) for mask in aperture_masks]
    for aperture, mask in zip(apertures, aperture_masks):
        if mask.shape != aperture.shape:
            raise ValueError("Each aperture mask must match its aperture shape.")

    if initial_aperture_fields is None:
        rng = np.random.default_rng(seed)
        phases = [
            rng.uniform(-np.pi, np.pi, aperture.shape)
            for aperture in apertures
        ]
    else:
        phases = [np.angle(field) for field in initial_aperture_fields]

    m = [np.zeros(aperture.shape, dtype=float) for aperture in apertures]
    v = [np.zeros(aperture.shape, dtype=float) for aperture in apertures]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    history = {"loss": [], "target_mean": [], "background_mean": []}

    for iteration in range(1, n_iter + 1):
        aperture_fields = [
            np.exp(1j * phase) * mask
            for phase, mask in zip(phases, aperture_masks)
        ]
        pressure = forward_pressure(
            points_m,
            apertures,
            aperture_fields,
            wavelength_m,
            target_extent_m,
            distance_bin_m=distance_bin_m,
            propagation_models=propagation_models,
        )
        loss, amp, aperture_grads = amplitude_loss_and_gradient(
            points_m,
            apertures,
            aperture_fields,
            pressure,
            desired_amp,
            loss_weights,
            pressure_scale,
            wavelength_m,
            target_extent_m,
            distance_bin_m=distance_bin_m,
            propagation_models=propagation_models,
        )

        for idx, (phase, field, grad_field) in enumerate(
            zip(phases, aperture_fields, aperture_grads)
        ):
            grad_phase = np.real(np.conj(grad_field) * (1j * field))
            m[idx] = beta1 * m[idx] + (1.0 - beta1) * grad_phase
            v[idx] = beta2 * v[idx] + (1.0 - beta2) * grad_phase**2
            m_hat = m[idx] / (1.0 - beta1**iteration)
            v_hat = v[idx] / (1.0 - beta2**iteration)
            phases[idx] = phase - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            phases[idx] = (phases[idx] + np.pi) % (2.0 * np.pi) - np.pi

        history["loss"].append(loss)
        history["target_mean"].append(float(amp[:target_count].mean()))
        history["background_mean"].append(float(amp[target_count:].mean()))

        if iteration % 20 == 0 or iteration == 1:
            print(
                f"iter {iteration:03d} loss={loss:.6e} "
                f"target_mean={history['target_mean'][-1]:.6e} "
                f"background_mean={history['background_mean'][-1]:.6e}"
            )

    return [
        np.exp(1j * phase) * mask
        for phase, mask in zip(phases, aperture_masks)
    ], history


def plot_results(
    x,
    y,
    z,
    pressure_amp,
    target_amp,
    apertures,
    aperture_fields,
    target_center_m,
    output_path,
    plot_pressure_percentile=99.5,
):
    n_phase_panels = len(apertures)
    fig = plt.figure(figsize=(5.0 + 4.0 * n_phase_panels, 6), constrained_layout=True)
    gs = fig.add_gridspec(
        1,
        1 + n_phase_panels,
        width_ratios=(1.35, *([1.0] * n_phase_panels)),
    )

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    threshold = np.percentile(pressure_amp, plot_pressure_percentile)
    hot = pressure_amp >= threshold
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    hot_points = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))[hot]
    hot_amp = pressure_amp[hot]
    scatter = ax3d.scatter(
        hot_points[:, 0] * 1e3,
        hot_points[:, 1] * 1e3,
        hot_points[:, 2] * 1e3,
        c=hot_amp,
        s=12,
        cmap="Reds",
        alpha=0.82,
        depthshade=False,
    )
    target_points = np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))[target_amp > 0.0]
    if target_points.size:
        ax3d.scatter(
            target_points[:, 0] * 1e3,
            target_points[:, 1] * 1e3,
            target_points[:, 2] * 1e3,
            c="cyan",
            marker=".",
            s=8,
            alpha=0.28,
            depthshade=False,
        )
    ax3d.scatter(
        [target_center_m[0] * 1e3],
        [target_center_m[1] * 1e3],
        [target_center_m[2] * 1e3],
        c="cyan",
        marker="x",
        s=80,
        linewidths=2,
    )
    ax3d.set_title("3D field strength")
    ax3d.set_xlabel("x [mm]")
    ax3d.set_ylabel("y [mm]")
    ax3d.set_zlabel("z [mm]")
    ax3d.set_xlim(x[0] * 1e3, x[-1] * 1e3)
    ax3d.set_ylim(y[0] * 1e3, y[-1] * 1e3)
    ax3d.set_zlim(z[0] * 1e3, z[-1] * 1e3)
    ax3d.view_init(elev=22, azim=-55)
    ax3d.set_box_aspect((1.2, 1.0, 1.0))
    fig.colorbar(scatter, ax=ax3d, label="|p|", shrink=0.68)

    for col, (aperture, aperture_field) in enumerate(
        zip(apertures, aperture_fields),
        start=1,
    ):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(
            aperture.uu_mm,
            aperture.vv_mm,
            np.angle(aperture_field),
            shading="auto",
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.set_aspect("equal")
        ax.set_title(f"{aperture.plane.name} phase")
        ax.set_xlabel("u [mm]")
        ax.set_ylabel("v [mm]")
        fig.colorbar(im, ax=ax, label="phase [rad]")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def assert_geometry(apertures):
    a, b = apertures
    checks = [
        np.allclose(a.plane.position_m, (0.0, 0.0, 0.0)),
        np.allclose(a.plane.normal, (1.0, 0.0, 0.0)),
        np.allclose(b.plane.position_m, (20e-3, 20e-3, 0.0)),
        np.allclose(b.plane.normal, (0.0, -1.0, 0.0)),
        np.isclose(a.width_m, 40e-3),
        np.isclose(b.width_m, 40e-3),
        np.isclose(np.dot(a.plane.normal, b.plane.normal), 0.0),
    ]
    if not all(checks):
        raise ValueError("PZT geometry check failed.")
