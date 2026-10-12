"""Paper-style field and geometry panels for the saved 1 kHz candidate."""
import json
import argparse
import os
from pathlib import Path
import numpy as np
from .geometry import Geometry, rasterize
from .solver import solve


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--spacing-mm',type=float,default=.5)
    parser.add_argument('--inspection-dir',type=Path,help='Reuse a saved inspect_candidate field without solving again')
    args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    root=Path(__file__).parent
    out=args.out or root/'results/field_presentation'
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.report or root/'results/target_1000_search/report.json').read_text())
    g=Geometry(**report['geometry'])
    if args.inspection_dir:
        from .solver import Result
        info=json.loads((args.inspection_dir/'inspection.json').read_text())
        g=Geometry(**info['geometry'])
        saved=np.load(args.inspection_dir/'field.npz')
        solid=saved['solid'];dx=float(saved['dx']);dz=float(saved['dz']);metrics=info['mesh']
        d=info['diagnostics']
        r=Result(saved['pressure'],d['reflection'],d['transmission'],d['energy_error'],d['residual'],
                 saved['outgoing_modes'],saved['kx'],saved['qz'],dx,dz)
        args.spacing_mm=dz*1000
    else:
        solid,dx,dz,metrics=rasterize(g,args.spacing_mm/1000)
        r=solve(solid,dx,dz,1000.)
    nx=solid.shape[1]
    x=(np.arange(nx)+.5)*dx
    basis=np.exp(1j*np.outer(x,r.kx))
    reflected=basis.conj().T@(r.pressure[0]-1)/nx
    distances=np.arange(dz,.5+dz/2,dz)
    # Negative z: incident wave advances right, reflected orders advance left.
    before=(np.exp(-1j*r.qz[0]*distances[:,None]/dz)
            + (np.exp(1j*np.outer(distances,r.qz/dz))*reflected)@basis.T)
    after=r.downstream(distances)
    field=np.concatenate([before[::-1],r.pressure,after])
    mask=np.concatenate([np.zeros_like(before,dtype=bool),solid,np.zeros_like(after,dtype=bool)])
    z=(np.arange(len(field))-len(before)+.5)*dz-.024
    extent=[(z[0]-dz/2)*1000,(z[-1]+dz/2)*1000,0,g.pitch*1000]
    length=metrics['total_structure_depth_m']*1000
    fig=plt.figure(figsize=(15,10),layout='constrained',facecolor='white')
    gs=fig.add_gridspec(3,2,width_ratios=[1,.028],height_ratios=[1.6,1,1])
    ax=fig.add_subplot(gs[0,0])
    # Crop to the actual structure and retain physical aspect ratio.
    active=np.flatnonzero(solid.any(axis=1))
    cropped=solid[active[0]:active[-1]+1]
    guide=metrics['actual_guide_length_m_each']*1000
    core=metrics['actual_core_depth_m']*1000
    width=g.pitch*1000
    ax.imshow(cropped.T,origin='lower',extent=[0,length,0,width],cmap='Greys',
              interpolation='nearest',vmin=0,vmax=1)
    ax.set(xlim=(-12,length+12),ylim=(-25,width+20),xlabel='Depth through wall (mm)',
           ylabel='Across repeating cell (mm)',title='(a) Selected metamaterial geometry — plan section')
    ax.axvline(guide,color='#087f8c',ls='--',lw=1)
    ax.axvline(guide+core,color='#087f8c',ls='--',lw=1)
    ax.text(guide/2,width+10,f'{guide:g} mm guide',ha='center',fontsize=10)
    ax.text(length/2,width+10,f'{core:g} mm ribbed core',ha='center',fontsize=10)
    ax.text(length-guide/2,width+10,f'{guide:g} mm guide',ha='center',fontsize=10)
    ax.text(length/2,width/2,f'{width*metrics["central_clear_fraction"]:g} mm clear passage',ha='center',va='center',fontsize=11)
    ax.text(length/2,-16,f'{length:g} mm total depth  •  {width:g} mm repeat width',ha='center',fontsize=10)
    for row,data,cmap,low,high,title,label in [
        (1,field.real,'RdBu_r',-2,2,'(b) Instantaneous pressure at 1,000 Hz — sound travels left to right',
         'Re(p) / incident amplitude'),
        (2,abs(field),'viridis',0,2,'(c) Pressure amplitude — distinguishes a pressure node from attenuation',
         '|p| / incident amplitude')]:
        ax=fig.add_subplot(gs[row,0]);cax=fig.add_subplot(gs[row,1])
        im=ax.imshow(np.ma.array(data.T,mask=mask.T),origin='lower',extent=extent,
                     cmap=cmap,vmin=low,vmax=high,interpolation='nearest',aspect='equal')
        overlay=np.ma.masked_where(~mask.T,np.ones(mask.T.shape))
        ax.imshow(overlay,origin='lower',extent=extent,cmap=ListedColormap(['#292d32']),
                  interpolation='nearest',aspect='equal')
        for pos in (0,length):
            ax.axvline(pos,color='#454545',lw=.6,ls='--')
        ax.set(title=title,xlabel='Propagation distance (mm; module starts at 0)',ylabel='Across cell (mm)')
        cb=fig.colorbar(im,cax=cax,extend='both' if row==1 else 'max');cb.set_label(label)
    tl=-10*np.log10(r.transmission)
    fig.suptitle(f'Optimised candidate, evaluated on a {args.spacing_mm:g} mm grid\n'
                 f'1,000 Hz · normal incidence · transmitted power {r.transmission:.1%} · '
                 f'reflected power {r.reflection:.1%} · transmission loss {tl:.2f} dB',fontsize=15)
    maximum=float(np.nanmax(abs(field)))
    fig.supxlabel(f'Lossless 2D periodic model; not validated outdoor performance. '
                  f'Colours clipped at ±2 / 2 for visibility; maximum |p| = {maximum:.2f}.\n'
                  f'{metrics["central_clear_fraction"]:.0%} central opening; '
                  f'{metrics["total_line_of_sight_fraction"]:.1%} total 2D sightline opening. '
                  'See the refinement report for mesh and mechanism checks.',fontsize=10)
    fig.savefig(out/'field_and_geometry.png',dpi=190)
    fig.savefig(out/'field_and_geometry.svg')
    plt.close(fig)
    np.savez_compressed(out/'field_1000hz_fine.npz',pressure=field,solid=mask,z_m=z,x_m=x,
                        frequency_hz=1000,dx=dx,dz=dz)
    info=dict(geometry=report['geometry'],mesh=metrics,transmission=r.transmission,
              reflection=r.reflection,tl_db=tl,energy_error=r.balance_error,
              maximum_pressure_amplitude=maximum)
    (out/'render_metrics.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
