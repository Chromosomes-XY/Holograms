"""Like-quadrature comparison of the selected 5 mm four/six cavity designs."""
import json,csv
import numpy as np
from .manufacturable_search import OUT,FC,POWER

def main():
    four=json.loads((OUT/'comparison_4_65.json').read_text());six=json.loads((OUT/'comparison_6_65.json').read_text())
    summary=dict(four_J=four['J'],six_J=six['J'],relative_transmitted_power_reduction=1-six['J']/four['J'],percentage_point_reduction=100*(four['J']-six['J']),extra_reduction_db=10*np.log10(four['J']/six['J']),four_depth_mm=four['depth_mm'],six_depth_mm=six['depth_mm'],caveat='Selected manufacturable four/six designs; depth differs. Not an isolated effect of adding two cavities; periodic normal-incidence power, not finite-wall receiver-level reduction.')
    (OUT/'four_vs_six_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    axs[1].semilogx(FC,POWER,'--',color='#888888',lw=1.7,label='Before wall: measured × A-weighting')
    for q,label,color in [(four,'4 cavities','#297bb4'),(six,'6 cavities','#149078')]:
        ts=[b['T'] if b['T'] is not None else np.nan for b in q['bands']]
        axs[0].semilogx(FC,ts,'o-',ms=4,lw=2,color=color,label=f'{label} · {q["depth_mm"]:.0f} mm depth')
        axs[1].semilogx(FC,[b['after'] for b in q['bands']],'o-',ms=4,lw=2,color=color,label=f'{label}: {100*q["J"]:.1f}% weighted power remains')
    for ax in axs:
        ax.set(xscale='log',xlim=(25,20000),xlabel='Frequency / band centre (Hz)');ax.grid(alpha=.2);ax.legend(fontsize=9)
        ticks=[25,63,125,250,500,1000,2000,4000,8000,16000];ax.set_xticks(ticks);ax.set_xticklabels([str(v) for v in ticks],rotation=35);ax.axvspan(800,1250,color='#dda94e',alpha=.10)
    axs[0].set(ylabel='Band-averaged power transmission',ylim=(0,1.04),title='Fraction of incident power transmitted')
    axs[1].set(ylabel='Normalised A-weighted band power',ylim=(0,1.05),title='Your noise spectrum after the wall')
    fig.suptitle(f'Four versus six covered cavities · {summary["extra_reduction_db"]:.2f} dB additional weighted reduction',fontsize=17)
    fig.supxlabel('Both: 300 mm pitch, 75% clear opening, 5 mm manufacturing grid. Shading: 800–1250 Hz.\nSame 2.5 mm mesh limit and 65-point dominant-band integration; all input bands contribute to totals.\nPeriodic, rigid, lossless 2D normal incidence. Different depths: this is a design comparison, not cavity count alone.',fontsize=10)
    fig.savefig(OUT/'four_vs_six_spectra.png',dpi=170);plt.close(fig)
    with (OUT/'four_vs_six_spectra.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['frequency_hz','weighted_before','four_transmission','six_transmission','four_weighted_after','six_weighted_after'])
        for i,(a,b) in enumerate(zip(four['bands'],six['bands'])):w.writerow([FC[i],POWER[i],a['T'],b['T'],a['after'],b['after']])
    print(json.dumps(summary,indent=2))
    for i,f in enumerate(FC):
        if f in [630,800,1000,1250,1600,2000]:print(f,four['bands'][i]['T'],six['bands'][i]['T'])

if __name__=='__main__':main()
