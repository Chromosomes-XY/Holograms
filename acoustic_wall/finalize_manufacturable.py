"""Manufacturing comparison and dimension schedules for the 5 mm study."""
import json,csv
import numpy as np
from .manufacturable_search import OUT,geometry,FC,POWER,evaluate

def main():
    r=json.loads((OUT/'report.json').read_text());notches=json.loads((OUT/'notch_report.json').read_text())
    qp=OUT/'winner_quadrature65.json'
    if not qp.exists():
        winner=min(r['selected'],key=lambda q:q['J'])
        qp.write_text(json.dumps(evaluate(winner['v'],n=9,h=.0025,dominant_n=65),indent=2))
    q65=json.loads(qp.read_text())
    for i,q in enumerate(r['selected']):
        if q['v']==q65['v']:
            q65.update(cavity_count=q['cavity_count'],J_9point=q['J_9point'],J_33point=q.get('J_33point',q['J']))
            r['selected'][i]=q65
    (OUT/'report.json').write_text(json.dumps(r,indent=2))
    entries=r['selected']+notches['selected'];best=min(r['selected'],key=lambda q:q['J']);simple=min(notches['selected'],key=lambda q:q['J'])
    for name,q,is_notch in [('covered_recess',best,False),('open_notch',simple,True)]:
        with (OUT/f'{name}_dimensions_mm.csv').open('w') as file:
            writer=csv.writer(file);writer.writerow(['name','side','start_from_front_mm','recess_length_mm','cut_penetration_from_side_mm','mouth_length_mm','cover_length_mm','cover_thickness_mm'])
            for c in q['cavities']:
                writer.writerow([c['name'],'left' if c['side']==0 else 'right',c['start'],c['length'],c['width']+5,c['length'] if is_notch else c['mouth'],0 if is_notch else c['length']-c['mouth'],0 if is_notch else 5])
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle,Patch
    fig=plt.figure(figsize=(14,10),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[1.4,.9,.9])
    ax=fig.add_subplot(gs[0,0]);ax2=fig.add_subplot(gs[0,1]);colours=['#247db0','#148c72','#a27046','#bcb077']
    labels=[]
    for q,col in zip(entries,colours):
        label=f'{q["cavity_count"]} '+('open notches' if q.get('notches') else 'covered recesses');labels.append(label)
        ax.semilogx(FC,[b['T'] if b['T'] is not None else np.nan for b in q['bands']],'o-',ms=3,color=col,label=label)
        ax2.semilogx(FC,[b['after'] for b in q['bands']],color=col,label=label)
    ax2.semilogx(FC,POWER,'--',color='#555',label='Before wall')
    for a in [ax,ax2]:a.set(xlim=(250,2500),xlabel='Nominal band centre (Hz)');a.grid(alpha=.2);a.legend(fontsize=9)
    ax.set(ylabel='Band-averaged power transmission',ylim=(0,1.04),title='All designs retain 75% clear opening')
    ax2.set(ylabel='Weighted band power',title='Your measured spectrum after attenuation')
    for row,(q,title,is_notch) in enumerate([(best,'Best covered-recess design',False),(simple,'Best completely open-notch design',True)],start=1):
        a=fig.add_subplot(gs[row,:]);s,x,z=geometry(q['v'],.0025,notches=is_notch)
        a.pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#edf4f5','#354450']),shading='flat')
        if not is_notch:
            for c in q['cavities']:
                a.add_patch(Rectangle((c['start']+c['mouth'],0 if c['side']==0 else 70),c['length']-c['mouth'],5,color='#df9a36'))
            a.legend(handles=[Patch(color='#df9a36',label='Separate flat cover strips · 5 mm thick')],loc='upper right',fontsize=9)
        a.set(xlim=(0,600),ylim=(-5,80),xlabel='Wall depth (mm)',ylabel='Slat width (mm)',title=f'{title}: {q["depth_mm"]:.0f} mm deep · {q["reduction_db"]:.2f} dB predicted weighted reduction')
        a.set_aspect('equal');a.grid(alpha=.15)
    fig.suptitle('5 mm manufacturing grid · rectangular cuts with or without cover strips',fontsize=17)
    fig.supxlabel('Cross-sections show only the opaque 75 mm slat; a 225 mm clear gap separates adjacent slats.\nNormal-incidence periodic 2D estimates, not finite-wall performance. Ideal square corners; attachment and sealing details are not designed.',fontsize=11)
    fig.savefig(OUT/'manufacturing_comparison.png',dpi=170);plt.close(fig)
    table='\n'.join(f'| {label} | {q["depth_mm"]:.0f} | {100*q["J"]:.1f}% | {q["reduction_db"]:.2f} |' for label,q in zip(labels,entries))
    text='''# Discrete 5 mm manufacturing study

All manufactured geometric lengths are integer multiples of 5 mm. The computational unit is a complete 75 mm slat plus 225 mm clear gap, eliminating the old 37.5 mm half-slat convention from the manufacturing geometry. Pitch is 300 mm and straight-through frontal clearance is exactly 75%. Opposing recesses leave at least a 5 mm central web; same-side recesses have at least 10 mm separation. Front and rear solid margins are at least 10 mm. Search depth is 350–600 mm, not an assertion that deeper designs cannot help.

## Results

| Construction | Depth (mm) | Weighted power remaining | Reduction (dB) |
|---|---:|---:|---:|
'''+table+'''

These are normal-incidence, lossless 2D periodic-unit estimates. They are not directly comparable with the preceding point-source finite-wall receiver-region dB results. All supplied nonzero bands enter the objective. Assume uniform power per Hz within the bands, supplied ear gain held at each band value, and incoherent addition of power. The zero-power 20 kHz input needs no solve.

## Construction interpretation

For the covered recess, cut a fully accessible rectangular recess from each side, then add a short flat 5 mm cover strip over part of its length. The exposed end is a rectangular mouth. This reproduces the simulated pocket without requiring a one-piece hidden undercut. Covers must close the intended cavity boundary; fasteners, seals, vertical end closures and structural details have not been engineered. The orange areas in the drawing identify these separate strips.

The open-notch option removes those strips completely and opens each recess over its full length. It is easier to access and clean, but it changes the acoustic geometry substantially. The result applies to this tested family of side-opening rectangular notches, not every possible notch/slot design. All corners are ideal square corners; actual cutter radii must be added before fabrication validation.

Dimension CSVs describe the selected covered-recess and open-notch options, including true cut penetration, mouth length and cover-strip size. Every dimension is a multiple of 5 mm. Plotting-mesh steps are numerical choices and are not manufacturing tolerances.

## Search and checks

40 discrete mutation proposals per four/six covered-recess family, and 20 per open-notch family, plus initial seeds. This is a limited local comparison, not a global optimum. Five-point Gauss integration screens candidates; rejected candidates may stop only when their accumulated nonnegative objective already exceeds the incumbent. Finalists use a finer 2.5 mm maximum mesh and nine points per band. Dominant 630–1600 Hz bands are subsequently checked with 33 points; the winning covered-recess design additionally uses 65. The rest retain nine points. Search meshes use 5 mm below 4 kHz upper band edge, otherwise 2.5 mm. High-frequency bands have very small weight but finite resolution remains a limitation.

Separate checks compare 2.5 and 1.25 mm meshes at identical nine-point integration in the 800, 1000 and 1250 Hz bands. The best covered and uncovered options are also screened at 0, ±30 and ±60 degrees, using five-point band integration. Those angular numbers are less frequency-resolved than the final normal-incidence score; compare like sampling when comparing angles. No directional distribution was assumed, and oblique incidence was not included in the search objective. See pocket_checks.json and notch_checks.json.

Reproduce from the repository root with:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.manufacturable_search
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.manufacturable_notches
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.check_manufacturable
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m acoustic_wall.finalize_manufacturable
```

The finalizer generates or reads winner_quadrature65.json. Caches are model-specific. The new geometries have not yet been run as a finite 9.9 m wall or with the intended 1.8 m height, realistic losses, source variation or weather exposure.
'''
    (OUT/'README.md').write_text(text)
    print(table)

if __name__=='__main__':main()
