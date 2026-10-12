"""Solve the fixed Figure 3 reconstruction scaled by two, without optimisation."""
import json, os
from pathlib import Path
import numpy as np
from .paper_benchmark import geometry, CASES
from .benchmark_mesh import PreparedMesh
from .independent_fem import solve_fem
OUT=Path(__file__).parent/'results'/'figure3_scaled_1000'/'refinement_attempts'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for edge in [.00005,.000025,.0000125]:
        s,x,z,m=geometry(CASES[0],.0003,'added','centred',guide=True,corner_step=edge)
        x=x*2;z=z*2
        mesh=PreparedMesh(s,x,z)
        fv=mesh.solve(frequency=1000)
        fem=solve_fem(s,x,z,frequency=1000);fp=fem.pop('pressure')
        row=dict(edge_mm=edge*2000,bulk_step_mm=.6,grid=list(s.shape),fv=dict(T=fv.transmission,R=fv.reflection,energy_error=fv.balance_error,residual=fv.residual),fem=fem)
        rows.append(row);print(json.dumps(row),flush=True)
        (OUT/'mesh_runs.json').write_text(json.dumps(rows,indent=2))
        np.savez_compressed(OUT/f'fields_edge{edge:g}.npz',solid=s,xedges=x,zedges=z,fv=fv.pressure,fem=fp)
        if abs(fem['T']-fv.transmission)>.02:
            raise RuntimeError('Full-device solvers disagree: saved diagnostic fields; no validated attenuation claim.')
    # Same topology and dimensionless mesh at 2 kHz: exact lossless scaling check.
    ref=PreparedMesh(s,x/2,z/2).solve(frequency=2000)
    scaling_error=abs(ref.transmission-fv.transmission)
    np.savez_compressed(OUT/'fields.npz',solid=s,xedges=x,zedges=z,fv=fv.pressure,fem=fp)
    report=dict(frequency_hz=1000,scale_factor=2,geometry=dict(core_mm=171.5,guide_each_mm=68.6,total_depth_mm=308.7,computational_width_mm=162,clear_channel_widths_mm=[50,50,50],side_slits_mm=[6.5,4],rib_axial_width_mm=10,wall_mm=4,rib_pairs_per_side=10,nominal_openness=1/3,central_clear_fraction=50/162,rib_placement='Equal pitch, inferred from diagram; not author CAD'),mesh_runs=rows,scaling_T_error=scaling_error,transmission_loss_db=float(-10*np.log10(fv.transmission)),limitations=['2D, rigid, lossless, normal incidence, rigid transverse exterior boundaries','Fixed reconstruction, not newly optimised and not 75% open','Axial rib placement remains unverified against author CAD'])
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    # Extend the FV field into the uniform rigid duct using its exact modal solution.
    d=np.linspace(0,.32,321)
    up=fv.extend(d,upstream=True);down=fv.extend(d)
    xc=(x[:-1]+x[1:])/2;zc=(z[:-1]+z[1:])/2
    fig,axs=plt.subplots(3,1,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[1,1,1]})
    for ax in axs:ax.set_facecolor('#30343b');ax.set_aspect('equal');ax.set_ylabel('Transverse x (mm)')
    axs[0].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#f3f7f8','#30343b']),vmin=0,vmax=1,shading='flat')
    axs[0].set_title('Geometry: 50 mm clear centre; two ribbed side channels; sound travels left → right',loc='left')
    field=np.ma.masked_where(s,fv.pressure.real)
    im=axs[1].pcolormesh(z*1000,x*1000,field.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat')
    axs[1].pcolormesh((zc[0]-d[::-1])*1000,xc*1000,up[::-1].real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest')
    axs[1].pcolormesh((zc[-1]+d)*1000,xc*1000,down.real.T,cmap='RdBu_r',vmin=-2,vmax=2,shading='nearest')
    axs[1].set_title(f'Finite volume · 1,000 Hz · T = {fv.transmission:.5f} · transmission loss = {report["transmission_loss_db"]:.2f} dB',loc='left')
    # FEM nodal pressure averaged to cell centres; align incident phase to FV port centre.
    aligned=fp*np.exp(-1j*2*np.pi*1000/343*(z[1]-z[0])/2)
    cells=(aligned[:-1,:-1]+aligned[1:,:-1]+aligned[:-1,1:]+aligned[1:,1:])/4
    axs[2].pcolormesh(z*1000,x*1000,np.ma.masked_where(s,cells.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat')
    axs[2].set_title(f'Independent triangular FEM · T = {fem["T"]:.5f} · same geometry and incident phase reference',loc='left')
    axs[2].set_xlabel('Axial position z (mm)')
    for ax in [axs[0],axs[2]]:ax.set_xlim(z[0]*1000,z[-1]*1000)
    fig.colorbar(im,ax=axs[1:],label='Re(p) / incident pressure (colour range ±2)',shrink=.8)
    fig.suptitle('Scaled Figure 3 reconstruction — computed acoustic pressure at 1,000 Hz',fontsize=16)
    fig.supxlabel('Rigid, lossless 2D duct model · 30.9% central clear fraction including walls · rib spacing inferred · not a 75%-open wall design',fontsize=10)
    fig.savefig(OUT/'pressure_1000.png',dpi=180);plt.close(fig)
    (OUT/'README.md').write_text(f'''# Fixed geometry at 1,000 Hz\n\nRun: `.venv/bin/python -m acoustic_wall.solve_scaled_1000`\n\nAll dimensions of the latest Figure 3 reconstruction are doubled, including guides, walls and inferred rib pitch. No optimisation was performed.\n\nFinite volume transmitted power fraction: {fv.transmission:.8f}; reflected fraction: {fv.reflection:.8f}; transmission loss: {report['transmission_loss_db']:.3f} dB. Independent FEM T: {fem['T']:.8f}. Absolute T difference: {abs(fem['T']-fv.transmission):.8f}. Scaling check against 2 kHz on the identical dimensionless mesh: {scaling_error:.3g}.\n\nSee report.json for three edge refinements, energy closure, dimensions and limitations; fields.npz contains both complex pressure fields. Figure shows instantaneous real pressure for unit incident pressure, not SPL. Colour saturates outside ±2. Exterior FV fields use modal continuation of the computed solution.\n\nThis is a rigid-sided, lossless, 2D normal-incidence calculation, not a finite outdoor wall prediction. Published clear channel and rib dimensions are used; equal axial rib spacing remains an assumption. Nominal central openness is one third; including walls, 50/162 = 30.86%. It does not meet the earlier 75% clear-view goal. It also does not resolve the previously documented discrepancy with the paper.\n''')
    print('DONE',str(OUT),flush=True)

if __name__=='__main__':main()
