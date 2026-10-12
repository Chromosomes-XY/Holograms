"""Plot measured phase/amplitude, diffraction orders and validation evidence."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    a=p.parse_args()
    data=json.loads(a.report.read_text());d=data['final_mesh_diagnostics']
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    x=np.arange(3);names=['Side A','Centre','Side C']
    measured=np.array(d['regional_amplitudes'])
    measured/=max(measured[[0,2]])
    axes[0,0].bar(x-.18,d['target_amplitudes'],.36,label='Ideal aperture target',color='#9ba6b4')
    axes[0,0].bar(x+.18,measured,.36,label='Assembled exit pressure',color='#147d92')
    axes[0,0].set(xticks=x,xticklabels=names,ylabel='Amplitude (largest side = 1)',title='Amplitude balance')
    axes[0,0].legend(fontsize=8)
    target=np.array(d['target_phase_deg']);raw=np.array(d['regional_phase_deg'])
    phase=target+np.rad2deg(np.angle(np.exp(1j*np.deg2rad(raw-target))))
    axes[0,1].plot(x,target,'o--',label='Paper phase prescription',color='#777777')
    axes[0,1].plot(x,phase,'o-',label='Assembled exit pressure',color='#147d92')
    axes[0,1].set(xticks=x,xticklabels=names,ylabel='Phase relative to centre (degrees)',title='Phase gradient')
    axes[0,1].legend(fontsize=8)
    for angle in (0,30,-30):
        rows=[r for r in data['checks'] if r['angle_deg']==angle]
        axes[0,2].plot([r['frequency_hz'] for r in rows],
                       [r['diagnostics']['tl_db'] for r in rows],'o-',label=f'{angle:g}°')
    axes[0,2].axvline(1000,color='gray',ls=':')
    axes[0,2].set(xlabel='Frequency (Hz)',ylabel='Transmission loss (dB)',title='0.5 mm mesh: frequency / angle checks')
    axes[0,2].legend(fontsize=8)
    modes=sorted([m for m in d['modes'] if abs(m['order'])<=5],key=lambda m:m['order'])
    axes[1,0].bar([m['order'] for m in modes],[m['pressure_amplitude'] for m in modes],
                  color=['#d07530' if m['propagating'] else '#147d92' for m in modes])
    axes[1,0].set(xlabel='Diffraction order',ylabel='Pressure amplitude at outlet',
                  title='Orange: propagating  |  teal: evanescent')
    dist=np.array(d['evanescent_rms_distance_m'])*1000
    axes[1,1].semilogy(dist,d['evanescent_pressure_rms'],'o-',label='Evanescent pressure RMS',color='#147d92')
    axes[1,1].axhline(max(d['propagating_pressure_rms'],1e-8),color='#d07530',ls='--',label='Propagating pressure RMS')
    axes[1,1].set(xlabel='Distance after outlet (mm)',ylabel='Pressure / incident amplitude',title='Near field decay (not a power fraction)')
    axes[1,1].legend(fontsize=8)
    refs=data['refinement']
    axes[1,2].plot([r['spacing_m']*1000 for r in refs],[r['diagnostics']['tl_db'] for r in refs],'o-',color='#147d92')
    axes[1,2].set(xlabel='Grid spacing (mm)',ylabel='Transmission loss at 1 kHz (dB)',title='Same nominal geometry, refined meshes')
    axes[1,2].invert_xaxis()
    for ax in axes.flat:
        ax.grid(alpha=.2,axis='y')
    fig.suptitle('Phase-guided reoptimisation — physical cell measurements and mechanism checks',fontsize=15)
    fig.supxlabel('Lossless rigid 2D periodic model. Ideal amplitude target uses a prescribed-pressure aperture surrogate.\n'
                  'Evanescent pressure does not carry outward power; these diagnostics do not establish a bound surface eigenmode.',fontsize=10)
    fig.savefig(a.report.parent/'phase_diagnostics.png',dpi=180)
    fig.savefig(a.report.parent/'phase_diagnostics.svg')
    plt.close(fig)


if __name__=='__main__':
    main()
