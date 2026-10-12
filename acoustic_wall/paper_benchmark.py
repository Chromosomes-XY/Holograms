"""Fixed published geometry and independently defined comparison cases.

Source: Yang et al 2025, Tables 1/2, Figures 3/5, Supplementary Notes 1/2.
No geometry fitting to the Figure 5 transmission curves is allowed here.
"""
from pathlib import Path
import numpy as np
from .benchmark_mesh import segmented_grid,graded_grid,PreparedMesh

ROOT=Path(__file__).parent/'benchmark'
# Table 2 order is the mirror of Figure 3's channel numbering. Preserve Table 2.
CASES=[
 dict(label=.33,side=25,centre=25,slits=[3.25,2],phase_pi=[.33,1,-.33],amplitude=[.69,1,.49],guide_ratio=.4),
 dict(label=.40,side=22.5,centre=30,slits=[2.61,1.78],phase_pi=[.30,1,-.30],amplitude=[.75,1,.56],guide_ratio=.3),
 dict(label=.47,side=20,centre=35,slits=[2.10,1.54],phase_pi=[.27,1,-.26],amplitude=[.79,1,.64],guide_ratio=.2),
 dict(label=.53,side=17.5,centre=40,slits=[1.63,1.30],phase_pi=[.23,1,-.22],amplitude=[.85,1,.73],guide_ratio=.1),
 dict(label=.60,side=15,centre=45,slits=[1.23,1.02],phase_pi=[.19,1,-.21],amplitude=[.89,1,.75],guide_ratio=.5),
 dict(label=.67,side=12.5,centre=50,slits=[.89,.78],phase_pi=[.18,1,-.16],amplitude=[.91,1,.84],guide_ratio=.5),
 dict(label=.73,side=10,centre=55,slits=[.58,.53],phase_pi=[.15,1,-.12],amplitude=[.93,1,.90],guide_ratio=.5)]


def geometry(case,step=.0005,width_convention='inclusive',rib_layout='centred',isolated=None,guide=True,corner_step=None):
    """Every rectangle edge is a mesh line. Width conventions are explicitly tested.

    inclusive: Wi spans between wall centrelines, clear Wi-tt, repeat W=75 mm.
    added: Wi is clear air, with tt=2 mm shared walls added (repeat 81 mm).
    Outer walls are represented by half thickness at each transverse end.
    h=5 mm is the axial rib dimension, tt=2 mm the longitudinal wall thickness.
    Rib end placement is not dimensioned by the article; two layouts are audited.
    """
    widths=np.array([case['side'],case['centre'],case['side']])*.001
    wall=.002;H=.08575;rib=.005;dH=case['guide_ratio']*H if guide else 0
    nominal=widths.copy();slits=np.array([case['slits'][0],case['centre'],case['slits'][1]])*.001
    if width_convention=='inclusive':clear=widths-wall
    elif width_convention=='added':clear=widths
    else:raise ValueError(width_convention)
    if isolated is not None:
        clear=np.array([clear[isolated]]);slits=np.array([slits[isolated]])
        ranges=[(0,float(clear[0]))];W=float(clear[0]);wall_intervals=[]
    else:
        start=wall/2;ranges=[];wall_intervals=[(0,wall/2)]
        for w in clear:
            ranges.append((start,start+w));start+=w+wall
        W=start-wall/2
        wall_intervals += [(ranges[j][1],ranges[j+1][0]) for j in range(2)]+[(ranges[-1][1],W)]
    length=H+2*dH
    rectangles=[(0,length,a,b) for a,b in wall_intervals]
    centres=((np.arange(10)+.5)*H/10 if rib_layout=='centred' else np.linspace(rib/2,H-rib/2,10))
    xbreak=[0,W];zbreak=[0,dH,dH+H,length]
    slit_breaks=[]
    for j,(a,b) in enumerate(ranges):
        xbreak.extend([a,b]);original=isolated if isolated is not None else j
        if original==1:continue
        slit=slits[j];lo=(a+b-slit)/2;hi=lo+slit
        # At least 6 transverse cells through each slit, preserving exact widths.
        xbreak.extend(np.linspace(lo,hi,7));slit_breaks.append((lo,hi))
        for centre in centres:
            z0=dH+centre-rib/2;z1=z0+rib
            rectangles.extend([(z0,z1,a,lo),(z0,z1,hi,b)]);zbreak.extend([z0,z1])
    grid=lambda points: segmented_grid(points,step) if corner_step is None else graded_grid(points,step,corner_step)
    xedges=grid(xbreak)
    # Uniform exterior spacing; independently vary padding for port checks.
    pad=.01;n=int(np.ceil(pad/step));hp=pad/n
    core=grid(zbreak)
    zedges=np.r_[np.linspace(-pad,0,n+1)[:-1],core,np.linspace(length,length+pad,n+1)[1:]]
    x=(xedges[1:]+xedges[:-1])/2;z=(zedges[1:]+zedges[:-1])/2
    solid=np.zeros((len(z),len(x)),bool)
    for z0,z1,a,b in rectangles:solid|=((z[:,None]>z0)&(z[:,None]<z1)&(x[None,:]>a)&(x[None,:]<b))
    meta=dict(case=case,step_m=step,corner_step_m=corner_step,width_convention=width_convention,rib_layout=rib_layout,
              width_m=W,length_m=length,core_m=H,guide_m=dH,clear_widths_m=clear.tolist(),
              nominal_openness=case['centre']/75,central_clear_fraction=float(clear[1]/W) if isolated is None else None,
              grid_shape=list(solid.shape),ranges_m=ranges,isolated=isolated,slits_m=slits.tolist())
    return solid,xedges,zedges,meta


def isolated_response(case,step=.0005,width_convention='inclusive',rib_layout='centred',corner_step=None):
    records=[]
    for i in [0,1,2]:
        solid,xe,ze,m=geometry(case,step,width_convention,rib_layout,isolated=i,guide=False,corner_step=corner_step)
        r=PreparedMesh(solid,xe,ze).solve()
        # Relative to empty propagation between the same numerical port centres.
        free_phase=r.q[0]/r.port_step*((ze[-1]+ze[-2]-ze[0]-ze[1])/2)
        t=r.forward[0]*np.exp(-1j*free_phase)
        records.append(dict(amplitude=float(abs(t)),relative_phase_deg=float(np.angle(t,deg=True)),
                            T=r.transmission,R=r.reflection,energy_error=r.balance_error,residual=r.residual))
    return records
