"""Figure-3-inspired rectangular ribbed-channel family, not a CAD replica.

The published nominal dimensions do not unambiguously include separator walls.
Here all walls are explicitly included in the pitch and frontal clear fraction.
z is the paper's propagation H axis; its t is the omitted extrusion direction.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Geometry:
    pitch: float = .200
    core_depth: float = .172
    guide_length: float = .068
    central_fraction: float = .75
    wall: float = .004
    rib_thickness: float = .010
    rib_count: int = 10
    slit_left: float = .006
    slit_right: float = .010
    offset_left: float = 0.
    offset_right: float = 0.

    def validate(self):
        vals = [v for k,v in asdict(self).items() if not k.startswith('offset_')]
        if not np.isfinite(vals).all() or min(vals) <= 0 or self.central_fraction >= 1:
            raise ValueError("Invalid geometry parameters")
        if not isinstance(self.rib_count, int):
            raise ValueError("rib_count must be an integer")
        side = (self.pitch*(1-self.central_fraction)-3*self.wall)/2
        if side <= max(self.slit_left, self.slit_right):
            raise ValueError("Side channel must be wider than its slit")
        if self.rib_count*self.rib_thickness >= self.core_depth:
            raise ValueError("Ribs leave no axial air spacing")
        if not np.isfinite([self.offset_left,self.offset_right]).all():
            raise ValueError('Offsets must be finite')
        if max(abs(self.offset_left),abs(self.offset_right)) > self.guide_length-.012:
            raise ValueError('Offsets must retain at least 12 mm entrance/exit guide')


def rasterize(g, spacing=.002, padding=.024):
    """Rasterize with cell-centred boundaries; return mask and actual mesh metrics."""
    g.validate()
    if spacing <= 0 or padding < 2*spacing:
        raise ValueError("Need positive spacing and at least two padding cells")
    nx = int(round(g.pitch/spacing))
    dx = g.pitch/nx
    dz = spacing
    wall = int(round(g.wall/dx))
    if wall < 2 or min(g.slit_left, g.slit_right, g.rib_thickness) < 2*spacing:
        raise ValueError("Resolve walls, slits and ribs with at least two cells")
    centre = int(np.ceil(g.central_fraction*nx-1e-10))
    left = (nx-centre-3*wall)//2
    right = nx-centre-3*wall-left
    if min(left, right) < 4:
        raise ValueError("Insufficient side-channel resolution")
    # Seam wall + left air + separator + central air + separator + right air.
    l0, l1 = wall, wall+left
    c0, c1 = l1+wall, l1+wall+centre
    r0, r1 = c1+wall, nx
    core_n = int(round(g.core_depth/dz))
    guide_n = int(round(g.guide_length/dz))
    pad_n = int(np.ceil(padding/dz))
    nz = 2*pad_n+2*guide_n+core_n
    solid = np.zeros((nz, nx), dtype=bool)
    region = slice(pad_n, nz-pad_n)
    for start, end in ((0,wall),(l1,c0),(c1,r0)):
        solid[region,start:end] = True
    thickness = max(2, int(round(g.rib_thickness/dz)))
    centres = pad_n+guide_n+(np.arange(g.rib_count)+.5)*core_n/g.rib_count
    for start, end, slit, offset in ((l0,l1,g.slit_left,g.offset_left),(r0,r1,g.slit_right,g.offset_right)):
        slit_n = max(2, int(round(slit/dx)))
        if end-start-slit_n < 2:
            raise ValueError("Need at least one cell per opposing rib protrusion")
        gap0 = start+(end-start-slit_n)//2
        for zc in centres:
            z0 = int(round(zc+offset/dz-thickness/2))
            solid[z0:z0+thickness,start:gap0] = True
            solid[z0:z0+thickness,gap0+slit_n:end] = True
    metrics = {
        "dx_m": dx, "dz_m": dz, "shape": [nz,nx],
        "central_clear_fraction": centre/nx,
        "total_line_of_sight_fraction": float((~solid.any(axis=0)).mean()),
        "actual_core_depth_m": core_n*dz,
        "actual_guide_length_m_each": guide_n*dz,
        "total_structure_depth_m": (core_n+2*guide_n)*dz,
        "note": "2D extruded model; top/bottom frames and posts are not included",
    }
    return solid, dx, dz, metrics


def paper_scaled_seed(target_hz=1000.):
    """Nominal Fig.3 proportions scaled from 2 kHz; walls added to clear widths.

    Explicit wall-inclusive pitch = (75+3*2)mm * scale. Thus this is a
    reconstruction seed, not an asserted reproduction of the published curve.
    """
    s = 2000./target_hz
    return Geometry(pitch=.081*s, core_depth=.08575*s, guide_length=.0343*s,
                    central_fraction=25/81, wall=.002*s, rib_thickness=.005*s,
                    slit_left=.002*s, slit_right=.00325*s)
