from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from hologramGen import (
    acoustic_params,
    export_hologram_stl,
    impedance_params,
    make_center_dot_target,
    make_grid,
    make_square_pzt,
    plot_hologram_outputs,
)


@dataclass(frozen=True)
class OrientedPlane:
    position_m: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray

    @classmethod
    def from_position_and_normal(cls, position_m, normal, tangent_hint=(0.0, 1.0, 0.0)):
        position_m = np.asarray(position_m, dtype=float)
        normal = unit_vector(normal)
        tangent_hint = np.asarray(tangent_hint, dtype=float)
        tangent_u = tangent_hint - np.dot(tangent_hint, normal) * normal
        if np.linalg.norm(tangent_u) < 1e-12:
            tangent_hint = np.array((0.0, 0.0, 1.0))
            tangent_u = tangent_hint - np.dot(tangent_hint, normal) * normal
        tangent_u = unit_vector(tangent_u)
        tangent_v = unit_vector(np.cross(normal, tangent_u))
        return cls(position_m, normal, tangent_u, tangent_v)


def unit_vector(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if norm == 0:
        raise ValueError("Vector must be non-zero.")
    return value / norm


def propagation_distance_m(source_plane, aim_plane):
    separation = aim_plane.position_m - source_plane.position_m
    off_axis = separation - np.dot(separation, source_plane.normal) * source_plane.normal
    normal_alignment = np.dot(source_plane.normal, aim_plane.normal)

    if np.linalg.norm(off_axis) > 1e-9:
        raise ValueError("This angular-spectrum model currently supports coaxial planes.")
    if normal_alignment < 0.999:
        raise ValueError("This angular-spectrum model currently supports parallel planes.")

    return float(np.dot(separation, source_plane.normal))


def asm_transfer(n, dx, lam, z):
    kx = 2 * np.pi * fftfreq(n, d=dx)
    kxv, kyv = np.meshgrid(kx, kx, indexing="xy")
    k = 2 * np.pi / lam
    k_perp2 = kxv**2 + kyv**2
    kz = np.sqrt(np.maximum(0.0, k**2 - k_perp2)) + 1j * np.sqrt(
        np.maximum(0.0, k_perp2 - k**2)
    )
    return np.exp(1j * z * kz)


def propagate_with_transfer(field, transfer):
    return ifft2(fft2(field) * transfer)


def adjoint_propagate_with_transfer(field_grad, transfer):
    return ifft2(fft2(field_grad) * np.conj(transfer))


def hologram_transfer(T, Zt, Zh, Zm, kh, km):
    T_eff = np.nan_to_num(T, nan=0.0)
    alpha_T = (
        4
        * Zt
        * Zh**2
        * Zm
        / (
            Zh**2 * (Zt + Zm) ** 2 * np.cos(kh * T_eff) ** 2
            + (Zh**2 + Zt * Zm) ** 2 * np.sin(kh * T_eff) ** 2
        )
    )
    return np.sqrt(alpha_T) * np.exp(1j * (km - kh) * T_eff)


def hologram_transfer_gradient(T, Zt, Zh, Zm, kh, km):
    step_m = 1e-8
    return (
        hologram_transfer(T + step_m, Zt, Zh, Zm, kh, km)
        - hologram_transfer(T - step_m, Zt, Zh, Zm, kh, km)
    ) / (2 * step_m)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def thickness_from_gamma(gamma, actuator_mask, Tmin, Tmax):
    sigma = sigmoid(gamma)
    T = Tmin + sigma * (Tmax - Tmin)
    T = np.where(actuator_mask, T, np.nan)
    dT_dgamma = (Tmax - Tmin) * sigma * (1.0 - sigma)
    return T, np.where(actuator_mask, dT_dgamma, 0.0)


def amplitude_mse_and_gradient(p_img, target_amp):
    eps = 1e-12
    amp = np.abs(p_img)
    residual = amp - target_amp
    loss = float(np.mean(residual**2))
    grad_img = (2.0 / p_img.size) * residual * p_img / (amp + eps)
    return loss, amp, grad_img


def centered_correlation(amp, target_amp):
    eps = 1e-12
    amp0 = amp - amp.mean()
    tgt0 = target_amp - target_amp.mean()
    return float(
        np.sum(amp0 * tgt0)
        / (np.sqrt(np.sum(amp0**2) * np.sum(tgt0**2)) + eps)
    )


def correlation_loss_and_gradient(p_img, target_amp, intensity_weight=0.0):
    eps = 1e-12
    amp = np.abs(p_img)
    amp0 = amp - amp.mean()
    tgt0 = target_amp - target_amp.mean()
    amp_norm = np.sqrt(np.sum(amp0**2)) + eps
    tgt_norm = np.sqrt(np.sum(tgt0**2)) + eps
    corr = np.sum(amp0 * tgt0) / (amp_norm * tgt_norm)

    d_corr_d_amp = (tgt0 / (amp_norm * tgt_norm)) - (
        corr * amp0 / (amp_norm**2)
    )
    d_loss_d_amp = -d_corr_d_amp

    if intensity_weight:
        intensity_norm = np.linalg.norm(amp) + eps
        d_loss_d_amp -= intensity_weight * amp / intensity_norm

    loss = float(-corr - intensity_weight * np.linalg.norm(amp))
    grad_img = d_loss_d_amp * p_img / (amp + eps)
    return loss, amp, grad_img


def design_hologram_gradient_descent(
    p0_amp,
    target_at_aim,
    source_plane,
    aim_plane,
    dx,
    lam_m,
    Zt,
    Zh,
    Zm,
    kh,
    km,
    n_iter=300,
    learning_rate=0.08,
    intensity_weight=0.0,
    Tmin=0.4e-3,
    Tmax=3e-3,
):
    actuator_mask = np.asarray(p0_amp) > 0
    p_in = np.asarray(p0_amp, dtype=np.complex128)
    target_amp = np.asarray(target_at_aim, dtype=float)
    transfer = asm_transfer(p0_amp.shape[0], dx, lam_m, propagation_distance_m(source_plane, aim_plane))

    initial_thickness = 0.5 * (Tmin + Tmax)
    initial_sigma = (initial_thickness - Tmin) / (Tmax - Tmin)
    gamma = np.full_like(p0_amp, np.log(initial_sigma / (1.0 - initial_sigma)), dtype=float)
    gamma = np.where(actuator_mask, gamma, 0.0)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = np.zeros_like(gamma)
    v = np.zeros_like(gamma)
    history = {"loss": [], "amp_mse": [], "amp_corr": [], "grad_norm": []}

    for iteration in range(1, n_iter + 1):
        T, dT_dgamma = thickness_from_gamma(gamma, actuator_mask, Tmin, Tmax)
        H = hologram_transfer(T, Zt, Zh, Zm, kh, km)
        p_exit = H * p_in
        p_img = propagate_with_transfer(p_exit, transfer)

        loss, amp, grad_img = correlation_loss_and_gradient(
            p_img,
            target_amp,
            intensity_weight=intensity_weight,
        )
        grad_exit = adjoint_propagate_with_transfer(grad_img, transfer)
        dH_dT = hologram_transfer_gradient(T, Zt, Zh, Zm, kh, km)
        grad_T = np.real(np.conj(grad_exit) * p_in * dH_dT)
        grad_gamma = np.where(actuator_mask, grad_T * dT_dgamma, 0.0)

        m = beta1 * m + (1.0 - beta1) * grad_gamma
        v = beta2 * v + (1.0 - beta2) * grad_gamma**2
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        gamma = gamma - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        gamma = np.where(actuator_mask, gamma, 0.0)

        history["loss"].append(loss)
        history["amp_mse"].append(float(np.mean((amp - target_amp) ** 2)))
        history["amp_corr"].append(centered_correlation(amp, target_amp))
        history["grad_norm"].append(float(np.linalg.norm(grad_gamma[actuator_mask])))

    T, _ = thickness_from_gamma(gamma, actuator_mask, Tmin, Tmax)
    return T, history


def project_hologram_from_poses(p0_amp, T, source_plane, aim_plane, dx, lam_m, Zt, Zh, Zm, kh, km):
    transfer = asm_transfer(p0_amp.shape[0], dx, lam_m, propagation_distance_m(source_plane, aim_plane))
    p_in = np.asarray(p0_amp, dtype=np.complex128)
    return propagate_with_transfer(hologram_transfer(T, Zt, Zh, Zm, kh, km) * p_in, transfer)


# Simulation grid
N = 512
SIMULATION_WIDTH_M = 0.045

# PZT and target
PZT_SIDE_M = 0.04
TARGET_DOT_DIAMETER_M = 0.1e-3

# 3D boundary/source and aim definitions.
SOURCE_PLANE = OrientedPlane.from_position_and_normal(
    position_m=(0.0, 0.0, 0.0),
    normal=(1.0, 0.0, 0.0),
)
AIM_PLANE = OrientedPlane.from_position_and_normal(
    position_m=(20e-3, 0.0, 0.0),
    normal=(1.0, 0.0, 0.0),
)

# Thickness caps applied inside the designer before export
MIN_THICKNESS_M = 0.4e-3
MAX_THICKNESS_M = 3.0e-3

# Material/acoustic parameters
FREQUENCY_HZ = 2.05e6
C_MEDIUM = 1500.0
C_HOLOGRAM = 2550.0

PROJECTED_DOMAIN_MM = (-0.5, 0.5, -2, 2)

PLOT_OUTPUT = SCRIPT_DIR / "square_pzt_hologram_gradient_output.png"
STL_OUTPUT = SCRIPT_DIR / "square_pzt_hologram_gradient_output.stl"


def main():
    _, xv, yv, dx = make_grid(N, SIMULATION_WIDTH_M)
    p0_amp = make_square_pzt(xv, yv, PZT_SIDE_M)
    target_amp = make_center_dot_target(xv, yv, TARGET_DOT_DIAMETER_M)

    acoustic = acoustic_params(
        frequency_hz=FREQUENCY_HZ,
        c_medium=C_MEDIUM,
        c_hologram=C_HOLOGRAM,
    )
    impedance = impedance_params(c_medium=C_MEDIUM, c_hologram=C_HOLOGRAM)
    aim_distance = propagation_distance_m(SOURCE_PLANE, AIM_PLANE)

    T, hist = design_hologram_gradient_descent(
        p0_amp=p0_amp,
        target_at_aim=target_amp,
        source_plane=SOURCE_PLANE,
        aim_plane=AIM_PLANE,
        dx=dx,
        lam_m=acoustic["lam_m"],
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
        n_iter=300,
        learning_rate=0.3,
        Tmin=MIN_THICKNESS_M,
        Tmax=MAX_THICKNESS_M,
    )

    p_img = project_hologram_from_poses(
        p0_amp=p0_amp,
        T=T,
        source_plane=SOURCE_PLANE,
        aim_plane=AIM_PLANE,
        dx=dx,
        lam_m=acoustic["lam_m"],
        Zt=impedance["Zt"],
        Zh=impedance["Zh"],
        Zm=impedance["Zm"],
        kh=acoustic["kh"],
        km=acoustic["km"],
    )

    n_facets, nonmanifold_edges = export_hologram_stl(T, dx, STL_OUTPUT)
    plot_hologram_outputs(
        xv=xv,
        yv=yv,
        T=T,
        p_img=p_img,
        z_target=aim_distance,
        output_path=PLOT_OUTPUT,
        projected_domain_mm=PROJECTED_DOMAIN_MM,
    )

    print("Source position [m]:", SOURCE_PLANE.position_m)
    print("Source normal:", SOURCE_PLANE.normal)
    print("Aim position [m]:", AIM_PLANE.position_m)
    print("Aim normal:", AIM_PLANE.normal)
    print("Aim distance along source normal [m]:", aim_distance)
    print("Final amp MSE:", hist["amp_mse"][-1])
    print("Final amp corr:", hist["amp_corr"][-1])
    print("Final gradient norm:", hist["grad_norm"][-1])
    print(f"Saved plot to {PLOT_OUTPUT}")
    print(f"Saved STL to {STL_OUTPUT} ({n_facets} facets)")
    print(f"STL nonmanifold edge count: {nonmanifold_edges}")


if __name__ == "__main__":
    main()
