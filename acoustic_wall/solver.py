"""Pressure Helmholtz scattering on a Cartesian finite-volume grid.

Convention exp(-i omega t), x transverse, z downstream, metres/Hz.
Rigid stair-step obstacles use zero normal flux. Bloch-periodic x boundaries
and exact *discrete* modal radiation conditions in z avoid sponge reflections.
This is not a thermoviscous, elastic, 3-D, or finite-wall solver.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class Result:
    pressure: np.ndarray
    reflection: float
    transmission: float
    balance_error: float
    residual: float
    outgoing_modes: np.ndarray
    kx: np.ndarray
    qz: np.ndarray
    dx: float
    dz: float
    transverse_basis: np.ndarray = None

    def downstream(self, distances):
        """1-D transverse angular spectrum; grid-consistent axial dispersion.

        Returns (distance, x). Uses outgoing modes only, not total field.
        Evanescent components decay. Distances start at the final grid centre.
        """
        distances = np.asarray(distances, dtype=float)
        if np.any(distances < 0):
            raise ValueError("Downstream distances must be nonnegative")
        x = (np.arange(len(self.kx)) + .5) * self.dx
        basis = (np.exp(1j * np.outer(x, self.kx)) if self.transverse_basis is None
                 else self.transverse_basis)
        return (np.exp(1j * np.outer(distances, self.qz / self.dz))
                * self.outgoing_modes) @ basis.T


def solve(solid, dx, dz, frequency=1000., angle_deg=0., c=343., transverse='periodic'):
    """Solve unit-pressure incidence from z<0; sum *all* propagating orders.

    solid is (nz,nx). First/last rows must be entirely air. Flux ratios use
    discrete group/flux factors sin(qz)/dz, consistent with the stencil.
    """
    solid = np.asarray(solid, dtype=bool)
    if solid.ndim != 2 or min(solid.shape) < 3:
        raise ValueError("Need a 2-D grid with at least three cells per axis")
    if not all(np.isfinite([dx, dz, frequency, angle_deg, c])):
        raise ValueError("Inputs must be finite")
    if min(dx, dz, frequency, c) <= 0 or abs(angle_deg) >= 85:
        raise ValueError("Positive scales required; incidence must be below 85 degrees")
    if solid[0].any() or solid[-1].any():
        raise ValueError("Radiation planes must lie entirely in air")
    nz, nx = solid.shape
    if transverse not in ('periodic', 'rigid'):
        raise ValueError("transverse must be periodic or rigid")
    if transverse == 'rigid' and angle_deg != 0:
        raise ValueError("Rigid channel ports currently support fundamental-mode incidence only")
    k = 2 * np.pi * frequency / c
    if k * max(dx, dz) > .6:
        raise ValueError("Grid too coarse: use at least about 11 points/wavelength")
    alpha = k * np.sin(np.deg2rad(angle_deg))
    width = nx * dx
    x = (np.arange(nx) + .5) * dx
    kx = (alpha + 2 * np.pi * np.fft.fftfreq(nx, d=dx) if transverse == 'periodic'
          else np.arange(nx)*np.pi/width)
    transverse_eigenvalue = (2 * np.sin(kx * dx / 2) / dx)**2
    qz = 2 * np.arcsin(.5 * dz * np.sqrt(k*k - transverse_eigenvalue + 0j))
    # Principal branch is outgoing for propagating modes and decaying otherwise.
    qz = np.where(qz.imag < 0, -qz, qz)
    roots = np.exp(1j * qz)
    basis = np.exp(1j * np.outer(x, kx)) / np.sqrt(nx)
    if transverse == 'rigid':
        basis = np.cos(np.outer(x,kx))*np.sqrt(2/nx)
        basis[:,0] /= np.sqrt(2)
    radiation = (basis * roots) @ basis.conj().T
    ids = np.full((nz, nx), -1, dtype=int)
    ids[~solid] = np.arange((~solid).sum())
    rows, cols, values = [], [], []
    rhs = np.zeros((~solid).sum(), dtype=complex)
    for iz, ix in zip(*np.where(~solid)):
        row = ids[iz, ix]
        diagonal = k*k
        for step in (-1, 1):
            if transverse == 'rigid' and not 0 <= ix+step < nx:
                continue
            jx = (ix + step) % nx
            if not solid[iz, jx]:
                phase = 1.
                if ix + step < 0:
                    phase = np.exp(-1j * alpha * width)
                elif ix + step >= nx:
                    phase = np.exp(1j * alpha * width)
                rows.append(row); cols.append(ids[iz, jx]); values.append(phase/dx**2)
                diagonal -= 1/dx**2
        for step in (-1, 1):
            jz = iz + step
            if 0 <= jz < nz:
                if not solid[jz, ix]:
                    rows.append(row); cols.append(ids[jz, ix]); values.append(1/dz**2)
                    diagonal -= 1/dz**2
            else:
                diagonal -= 1/dz**2
                rows.extend([row]*nx); cols.extend(ids[iz].tolist())
                values.extend((radiation[ix]/dz**2).tolist())
                if iz == 0:
                    rhs[row] = -(1/roots[0]-roots[0])*np.exp(1j*alpha*x[ix])/dz**2
        rows.append(row); cols.append(row); values.append(diagonal)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(rhs), len(rhs))).tocsc()
    p = spsolve(matrix, rhs)
    if not np.isfinite(p).all():
        raise RuntimeError("Nonfinite solution (possibly disconnected resonant cavity)")
    residual = float(np.linalg.norm(matrix@p-rhs)/np.linalg.norm(rhs))
    field = np.full(solid.shape, np.nan+0j)
    field[~solid] = p
    reflected = basis.conj().T @ (field[0]-np.exp(1j*alpha*x))/np.sqrt(nx)
    transmitted = basis.conj().T @ field[-1]/np.sqrt(nx)
    flux = np.where(abs(qz.imag) < 1e-10, np.sin(qz.real)/dz, 0.)
    weights = flux / flux[0]
    r = float(np.sum(weights * abs(reflected)**2))
    t = float(np.sum(weights * abs(transmitted)**2))
    return Result(field, r, t, abs(r+t-1), residual, transmitted, kx, qz, dx, dz,
                  basis*np.sqrt(nx))
