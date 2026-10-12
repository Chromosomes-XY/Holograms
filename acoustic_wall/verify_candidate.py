"""Mesh and off-design checks for a saved candidate (not physical validation)."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import numpy as np
from .geometry import Geometry
from .run import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.report.read_text())
    g=Geometry(**source['geometry'])
    refinement=[]
    for spacing in (.002,.001,.0005):
        rows,mesh=evaluate(g,[1000.],[0.,30.,-30.],spacing)
        refinement.append(dict(spacing_m=spacing,mesh=mesh,rows=rows))
        print(f'mesh {spacing*1000:g} mm: '+str([round(r['tl_db'],3) for r in rows]),flush=True)
    sweep,_=evaluate(g,list(np.arange(800.,1251.,25.)),[0.,30.,-30.],.001)
    perturbations=[]
    for name in ('slit_left','slit_right','core_depth','guide_length'):
        for delta in (-.001,.001):
            candidate=replace(g,**{name:getattr(g,name)+delta})
            rows,_=evaluate(candidate,[1000.],[0.],.001)
            perturbations.append(dict(parameter=name,delta_m=delta,rows=rows))
    differences=[abs(a['transmission']-b['transmission'])
                 for a,b in zip(refinement[-1]['rows'],refinement[-2]['rows'])]
    report=dict(source=str(args.report),geometry=source['geometry'],refinement=refinement,
                max_fine_transmission_change=max(differences),
                provisional_convergence_pass=max(differences)<.02,
                convergence_rule='absolute T difference < 0.02 between 1 and 0.5 mm; not a rigorous error bound',
                spectrum=sweep,perturbations=perturbations,
                warning='lossless 2D screening only; no paper reproduction or experimental validation')
    (args.out/'verification.json').write_text(json.dumps(report,indent=2))
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for a in (0.,30.,-30.):
        r=[r for r in sweep if r['angle_deg']==a]
        ax.plot([v['frequency_hz'] for v in r],[v['tl_db'] for v in r],label=f'{a:g}°',alpha=.8)
    ax.axvline(1000,color='black',linestyle=':',label='Target')
    ax.set(xlabel='Frequency (Hz)',ylabel='Transmission loss (dB)',
           title='Candidate checked on 1 mm mesh — lossless periodic 2D model')
    ax.legend();ax.grid(alpha=.25)
    fig.savefig(args.out/'spectrum.png',dpi=170);plt.close(fig)
    print(json.dumps({k:report[k] for k in ['max_fine_transmission_change','provisional_convergence_pass']},indent=2))


if __name__=='__main__':
    main()
