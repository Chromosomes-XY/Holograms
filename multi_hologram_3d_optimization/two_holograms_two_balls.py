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


def make_volume_grid():
    x = np.linspace(16e-3, 28e-3, 32)
    y = np.linspace(-8e-3, 8e-3, 26)
    z = np.linspace(-8e-3, 8e-3, 26)
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))


def make_ball_target(points_m, centers_m, radius_m):
    target = np.zeros(points_m.shape[0], dtype=float)
    for center in centers_m:
        r = np.linalg.norm(points_m - np.asarray(center), axis=1)
        target = np.maximum(target, (r <= radius_m).astype(float))
    return target


def make_optimization_samples(points_m, target_amp, max_background, seed=11):
    rng = np.random.default_rng(seed)
    target_idx = np.flatnonzero(target_amp > 0.0)
    background_idx = np.flatnonzero(target_amp <= 0.0)
    if background_idx.size > max_background:
        background_idx = rng.choice(background_idx, size=max_background, replace=False)
    sample_idx = np.concatenate((target_idx, background_idx))
    desired_amp = np.zeros(sample_idx.size, dtype=float)
    weights = np.full(sample_idx.size, 0.2, dtype=float)
    desired_amp[: target_idx.size] = 1.0
    weights[: target_idx.size] = 18.0
    return sample_idx, points_m[sample_idx], desired_amp, weights, target_idx.size


def point_local_coordinates(points_m, plane):
    rel = points_m - plane.position_m[None, :]
    u = rel @ plane.tangent_u
    v = rel @ plane.tangent_v
    s = rel @ plane.normal
    return u, v, s


def angular_spectrum_transfer(aperture, wavelength_m, distance_m, target_extent_m):
    key = round(float(distance_m), 12)
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


def propagate_aperture(aperture_field, aperture, wavelength_m, distance_m, target_extent_m):
    transfer = angular_spectrum_transfer(
        aperture,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
        target_extent_m=target_extent_m,
    )
    return ifft2(fft2(aperture_field) * transfer)


def adjoint_propagate(field_grad, aperture, wavelength_m, distance_m, target_extent_m):
    transfer = angular_spectrum_transfer(
        aperture,
        wavelength_m=wavelength_m,
        distance_m=distance_m,
        target_extent_m=target_extent_m,
    )
    return ifft2(fft2(field_grad) * np.conj(transfer))


def bilinear_sample(field, coords_m, u, v):
    dx = coords_m[1] - coords_m[0]
    u_idx = (u - coords_m[0]) / dx
    v_idx = (v - coords_m[0]) / dx
    j0 = np.floor(u_idx).astype(int)
    i0 = np.floor(v_idx).astype(int)
    du = u_idx - j0
    dv = v_idx - i0
    valid = (
        (i0 >= 0)
        & (j0 >= 0)
        & (i0 < field.shape[0] - 1)
        & (j0 < field.shape[1] - 1)
    )
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


def forward_pressure(points_m, apertures, aperture_fields, wavelength_m, target_extent_m):
    pressure = np.zeros(points_m.shape[0], dtype=np.complex128)
    for aperture, aperture_field in zip(apertures, aperture_fields):
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        forward = s > 0
        for distance in np.unique(np.round(s[forward], 12)):
            slice_mask = forward & np.isclose(s, distance, atol=1e-12)
            propagated = propagate_aperture(
                aperture_field,
                aperture,
                wavelength_m,
                float(distance),
                target_extent_m,
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
):
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
    for aperture in apertures:
        grad_aperture = np.zeros(aperture.shape, dtype=np.complex128)
        u, v, s = point_local_coordinates(points_m, aperture.plane)
        forward = s > 0
        for distance in np.unique(np.round(s[forward], 12)):
            slice_mask = forward & np.isclose(s, distance, atol=1e-12)
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
            )
        aperture_grads.append(grad_aperture)

    return loss, amp, aperture_grads


def project_unit_phase(aperture_fields):
    return [np.exp(1j * np.angle(field)) for field in aperture_fields]


def initialize_from_focus_points(apertures, focus_points_m, wavelength_m, target_extent_m):
    aperture_fields = []
    for aperture in apertures:
        seed_field = np.zeros(aperture.shape, dtype=np.complex128)
        u, v, s = point_local_coordinates(np.asarray(focus_points_m), aperture.plane)
        for distance in np.unique(np.round(s[s > 0], 12)):
            point_mask = (s > 0) & np.isclose(s, distance, atol=1e-12)
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
            )
        if np.max(np.abs(seed_field)) < 1e-12:
            seed_field = np.ones(aperture.shape, dtype=np.complex128)
        aperture_fields.append(np.exp(1j * np.angle(seed_field)))
    return aperture_fields


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
):
    pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m,
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


def plot_results(
    x,
    y,
    z,
    pressure_amp,
    target_amp,
    apertures,
    aperture_fields,
    ball_centers_m,
    output_path,
):
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.35, 1.0, 1.0))

    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    threshold = np.percentile(pressure_amp, 97.5)
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
    for center in ball_centers_m:
        ax3d.scatter(
            [center[0] * 1e3],
            [center[1] * 1e3],
            [center[2] * 1e3],
            c="cyan",
            marker="x",
            s=80,
            linewidths=2,
        )
    ax3d.set_title("3D field strength")
    ax3d.set_xlabel("x [mm]")
    ax3d.set_ylabel("y [mm]")
    ax3d.set_zlabel("z [mm]")
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


def main():
    frequency_hz = 2.05e6
    sound_speed_m_s = 1500.0
    wavelength_m = sound_speed_m_s / frequency_hz

    ball_centers_m = [
        np.array((20e-3, 0.0, 0.0)),
        np.array((24e-3, 0.0, 0.0)),
    ]
    ball_radius_m = 0.75e-3

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

    pzt_side_m = 40e-3
    pzt_n_side = 64
    apertures = [
        make_hologram_aperture(plane, pzt_n_side, pzt_side_m)
        for plane in pzt_planes
    ]
    assert_geometry(apertures)

    x, y, z, points_m = make_volume_grid()
    target_amp = make_ball_target(points_m, ball_centers_m, ball_radius_m)
    target_pressure_level = 1.0
    _, sample_points_m, desired_amp, weights, target_count = make_optimization_samples(
        points_m,
        target_amp,
        max_background=4500,
    )

    print("Optimizing two 40 mm square orthogonal PZTs for two 3D pressure balls")
    for aperture in apertures:
        print(
            f"{aperture.plane.name} position [m], normal, side [m]:",
            aperture.plane.position_m,
            aperture.plane.normal,
            aperture.width_m,
        )
    print("PZT normal dot product:", float(np.dot(pzt_planes[0].normal, pzt_planes[1].normal)))
    print("Ball centers [m]:", ball_centers_m)
    print("Volume sample count:", points_m.shape[0])
    print("Optimization sample count:", sample_points_m.shape[0])
    print("PZT phase pixels:", sum(aperture.size for aperture in apertures))

    initial_aperture_fields = initialize_from_focus_points(
        apertures,
        ball_centers_m,
        wavelength_m,
        target_extent_m=max(y[-1] - y[0], z[-1] - z[0]),
    )
    _, initial_amp, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        initial_aperture_fields,
        desired_amp,
        weights,
        target_pressure_level,
        wavelength_m,
        target_extent_m=max(y[-1] - y[0], z[-1] - z[0]),
    )
    target_pressure_level = float(initial_amp[:target_count].mean())
    initial_loss, initial_amp, _ = evaluate_amplitude_loss(
        sample_points_m,
        apertures,
        initial_aperture_fields,
        desired_amp,
        weights,
        target_pressure_level,
        wavelength_m,
        target_extent_m=max(y[-1] - y[0], z[-1] - z[0]),
    )
    print("Initial focused-field loss:", initial_loss)
    print("Initial focused target mean:", float(initial_amp[:target_count].mean()))
    print("Initial focused background mean:", float(initial_amp[target_count:].mean()))

    aperture_fields, history = optimize_projected_phase(
        points_m=sample_points_m,
        apertures=apertures,
        desired_amp=desired_amp,
        loss_weights=weights,
        pressure_scale=target_pressure_level,
        target_count=target_count,
        wavelength_m=wavelength_m,
        target_extent_m=max(y[-1] - y[0], z[-1] - z[0]),
        n_iter=80,
        initial_step=2.0,
        max_step=32.0,
        beta=0.5,
        sigma=0.1,
        initial_aperture_fields=initial_aperture_fields,
    )

    pressure = forward_pressure(
        points_m,
        apertures,
        aperture_fields,
        wavelength_m,
        target_extent_m=max(y[-1] - y[0], z[-1] - z[0]),
    )
    pressure_amp = np.abs(pressure)
    output_path = SCRIPT_DIR / "two_holograms_two_balls_pressure.png"
    plot_results(
        x=x,
        y=y,
        z=z,
        pressure_amp=pressure_amp,
        target_amp=target_amp,
        apertures=apertures,
        aperture_fields=aperture_fields,
        ball_centers_m=ball_centers_m,
        output_path=output_path,
    )

    target_mask = target_amp > 0.0
    print("Final loss:", history["loss"][-1])
    print("Final mean pressure inside balls:", float(pressure_amp[target_mask].mean()))
    print("Final mean pressure outside balls:", float(pressure_amp[~target_mask].mean()))
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
