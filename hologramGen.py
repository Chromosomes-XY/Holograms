import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from numpy.fft import fft2, ifft2, fftfreq


def make_grid(n, width_m):
    dx = width_m / n
    x = (np.arange(n) - n / 2) * dx
    xv, yv = np.meshgrid(x, x, indexing="xy")
    return x, xv, yv, dx


def make_square_pzt(xv, yv, side_m):
    half_side = side_m / 2
    return ((np.abs(xv) <= half_side) & (np.abs(yv) <= half_side)).astype(float)


def make_center_dot_target(xv, yv, diameter_m):
    r = np.sqrt(xv**2 + yv**2)
    return (r <= diameter_m / 2).astype(float)


def acoustic_params(frequency_hz, c_medium=1500.0, c_hologram=2550.0):
    lam_m = c_medium / frequency_hz
    lam_h = c_hologram / frequency_hz
    return {
        "lam_m": lam_m,
        "km": 2 * np.pi / lam_m,
        "kh": 2 * np.pi / lam_h,
    }


def impedance_params(
    rho_medium=1000.0,
    rho_hologram=1200.0,
    rho_transducer=7800.0,
    c_medium=1500.0,
    c_hologram=2550.0,
    c_transducer=5900.0,
):
    return {
        "Zm": rho_medium * c_medium,
        "Zh": rho_hologram * c_hologram,
        "Zt": rho_transducer * c_transducer,
    }


def propagate_ASM(U0, dx, lam, z):
    n = U0.shape[0]
    A = fft2(U0)

    kx = 2 * np.pi * fftfreq(n, d=dx)
    kxv, kyv = np.meshgrid(kx, kx, indexing="xy")

    k = 2 * np.pi / lam
    k_perp2 = kxv**2 + kyv**2

    kz = np.sqrt(np.maximum(0.0, k**2 - k_perp2)) + 1j * np.sqrt(
        np.maximum(0.0, k_perp2 - k**2)
    )

    if z < 0:
        kz = np.real(kz)

    return ifft2(A * np.exp(1j * z * kz))


def apply_hologram(p_in, T, Zt, Zh, Zm, kh, km):
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

    H = np.sqrt(alpha_T) * np.exp(1j * (km - kh) * T_eff)
    return H * p_in


def design_hologram_single_plane(
    p0_amp,
    target_at_z,
    z_target,
    dx,
    lam_m,
    Zt,
    Zh,
    Zm,
    kh,
    km,
    n_iter=300,
    Tmin=0.4e-3,
    Tmax=3e-3,
):
    actuator_mask = np.asarray(p0_amp) > 0
    p_in = np.asarray(p0_amp, dtype=np.complex128)

    T = np.zeros_like(p0_amp, dtype=float)
    T[actuator_mask] = Tmin
    T[~actuator_mask] = np.nan

    A_tgt = np.asarray(target_at_z, dtype=float)
    eps = 1e-12
    history = {"amp_mse": [], "amp_corr": []}

    for _ in range(n_iter):
        p_exit = apply_hologram(p_in, T, Zt, Zh, Zm, kh, km)
        p_img = propagate_ASM(p_exit, dx=dx, lam=lam_m, z=z_target)

        A = np.abs(p_img)
        mse = np.mean((A - A_tgt) ** 2)
        num = np.sum((A - A.mean()) * (A_tgt - A_tgt.mean()))
        den = np.sqrt(
            np.sum((A - A.mean()) ** 2) * np.sum((A_tgt - A_tgt.mean()) ** 2)
            + eps
        )
        history["amp_mse"].append(float(mse))
        history["amp_corr"].append(float(num / (den + eps)))

        p_img = A_tgt * np.exp(1j * np.angle(p_img))
        p_exit_back = propagate_ASM(p_img, dx=dx, lam=lam_m, z=-z_target)

        H_req = p_exit_back / (p_in + eps)
        dK = km - kh
        phase = np.angle(H_req)
        T = np.mod(phase, 2 * np.pi) / (dK + eps)
        T[actuator_mask] = np.clip(T[actuator_mask], Tmin, Tmax)
        T[~actuator_mask] = np.nan

    return T, history


def project_hologram(p0_amp, T, dx, lam_m, z_target, Zt, Zh, Zm, kh, km):
    p_in = np.asarray(p0_amp, dtype=np.complex128)
    p_exit = apply_hologram(p_in, T, Zt, Zh, Zm, kh, km)
    return propagate_ASM(p_exit, dx=dx, lam=lam_m, z=z_target)


def facet_normal(v1, v2, v3):
    normal = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return np.zeros(3)
    return normal / norm


def write_ascii_stl(path, triangles, solid_name="hologram"):
    with open(path, "w", encoding="ascii") as f:
        f.write(f"solid {solid_name}\n")
        for tri in triangles:
            v1, v2, v3 = tri
            normal = facet_normal(v1, v2, v3)
            f.write(
                f"  facet normal {normal[0]:.9e} {normal[1]:.9e} {normal[2]:.9e}\n"
            )
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(
                    f"      vertex {vertex[0]:.9e} {vertex[1]:.9e} {vertex[2]:.9e}\n"
                )
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {solid_name}\n")


def build_stl_triangles(T, dx):
    height_mm = np.asarray(T, dtype=float) * 1e3
    valid_vertices = np.isfinite(height_mm)
    n = height_mm.shape[0]

    x_mm = (np.arange(n) - n / 2) * dx * 1e3
    y_mm = (np.arange(n) - n / 2) * dx * 1e3

    top = np.zeros((n, n, 3), dtype=float)
    bottom = np.zeros((n, n, 3), dtype=float)
    for i, y in enumerate(y_mm):
        for j, x in enumerate(x_mm):
            z = height_mm[i, j] if valid_vertices[i, j] else 0.0
            top[i, j] = (x, y, z)
            bottom[i, j] = (x, y, 0.0)

    valid_cells = (
        valid_vertices[:-1, :-1]
        & valid_vertices[1:, :-1]
        & valid_vertices[:-1, 1:]
        & valid_vertices[1:, 1:]
    )

    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            if not valid_cells[i, j]:
                continue

            tl = top[i, j]
            tr = top[i, j + 1]
            bl = top[i + 1, j]
            br = top[i + 1, j + 1]
            btl = bottom[i, j]
            btr = bottom[i, j + 1]
            bbl = bottom[i + 1, j]
            bbr = bottom[i + 1, j + 1]

            triangles.append((tl, tr, bl))
            triangles.append((tr, br, bl))

            triangles.append((btl, bbl, btr))
            triangles.append((btr, bbl, bbr))

            if i == 0 or not valid_cells[i - 1, j]:
                triangles.append((tl, btl, tr))
                triangles.append((tr, btl, btr))
            if i == n - 2 or not valid_cells[i + 1, j]:
                triangles.append((bl, br, bbl))
                triangles.append((br, bbr, bbl))
            if j == 0 or not valid_cells[i, j - 1]:
                triangles.append((tl, bl, btl))
                triangles.append((bl, bbl, btl))
            if j == n - 2 or not valid_cells[i, j + 1]:
                triangles.append((tr, btr, br))
                triangles.append((br, btr, bbr))

    return triangles


def count_nonmanifold_edges(triangles, precision=6):
    edges = Counter()
    for tri in triangles:
        rounded = [tuple(round(float(v), precision) for v in vertex) for vertex in tri]
        for a, b in ((rounded[0], rounded[1]), (rounded[1], rounded[2]), (rounded[2], rounded[0])):
            edges[tuple(sorted((a, b)))] += 1
    return sum(1 for count in edges.values() if count != 2)


def export_hologram_stl(T, dx, output_path="hologram_output.stl"):
    triangles = build_stl_triangles(T, dx)
    write_ascii_stl(output_path, triangles)
    return len(triangles), count_nonmanifold_edges(triangles)


def plot_hologram_outputs(
    xv,
    yv,
    T,
    p_img,
    z_target,
    output_path="hologram_output.png",
    projected_domain_mm=None,
):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    im0 = axs[0].pcolormesh(xv * 1e3, yv * 1e3, np.abs(T) * 1e3, shading="auto")
    axs[0].set_aspect("equal")
    axs[0].set_title("Thickness field")
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    fig.colorbar(im0, ax=axs[0], label="thickness [mm]")

    im1 = axs[1].pcolormesh(
        xv * 1e3, yv * 1e3, np.abs(p_img), shading="auto", cmap="inferno"
    )
    axs[1].set_aspect("equal")
    axs[1].set_title(f"Projected image at z = {z_target * 1e3:.0f} mm")
    axs[1].set_xlabel("x [mm]")
    axs[1].set_ylabel("y [mm]")
    if projected_domain_mm is not None:
        xmin, xmax, ymin, ymax = projected_domain_mm
        axs[1].set_xlim(xmin, xmax)
        axs[1].set_ylim(ymin, ymax)
    fig.colorbar(im1, ax=axs[1], label="amplitude")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
