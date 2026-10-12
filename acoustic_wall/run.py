"""Run from repository root: python -m acoustic_wall.run --help."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time
import numpy as np
from scipy.optimize import differential_evolution
from .geometry import Geometry, rasterize, paper_scaled_seed
from .solver import solve


def numbers(s):
    return [float(v) for v in s.split(',')]


def evaluate(g, frequencies, angles, spacing):
    mask, dx, dz, metrics = rasterize(g, spacing)
    rows = []
    for f in frequencies:
        for a in angles:
            result = solve(mask, dx, dz, frequency=f, angle_deg=a)
            if result.balance_error > 1e-6 or result.residual > 1e-7:
                raise RuntimeError("Scattering solve failed conservation/residual check")
            rows.append(dict(frequency_hz=f, angle_deg=a,
                             transmission=result.transmission, reflection=result.reflection,
                             tl_db=float(-10*np.log10(max(result.transmission,1e-15))),
                             energy_error=result.balance_error, residual=result.residual))
    return rows, metrics


def score(rows):
    t = np.array([r['transmission'] for r in rows])
    return float(.7*t.mean()+.3*t.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['evaluate','optimise'])
    p.add_argument('--out', type=Path, default=Path('acoustic_wall/results/demo'))
    p.add_argument('--spacing-mm', type=float, default=2.)
    p.add_argument('--frequencies', type=numbers, default=[900.,1000.,1100.])
    p.add_argument('--angles', type=numbers, default=[0.])
    p.add_argument('--iterations', type=int, default=3)
    p.add_argument('--popsize', type=int, default=4)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--paper-seed', action='store_true')
    p.add_argument('--geometry', type=Path, help='JSON containing a geometry object')
    args = p.parse_args()
    if not args.frequencies or not args.angles:
        p.error('At least one frequency and angle required')
    args.out.mkdir(parents=True, exist_ok=True)
    g = paper_scaled_seed() if args.paper_seed else Geometry()
    if args.geometry:
        g = Geometry(**json.loads(args.geometry.read_text())['geometry'])
    if args.paper_seed and args.mode == 'optimise':
        p.error('Paper seed is a reconstruction baseline; optimise the 75% family instead')
    spacing = args.spacing_mm/1000
    start = time.monotonic()
    baseline, metrics = evaluate(g, args.frequencies, args.angles, spacing)
    initial_geometry = asdict(g)
    history = []
    if args.mode == 'optimise':
        best = [score(baseline),g]
        cache = {}
        # Lattice snapping matches the Cartesian raster. No fictitious optimisation
        # of independently assigned phase/amplitude; every candidate is geometry.
        def objective(v):
            v = tuple(float(round(x/spacing)*spacing) for x in v)
            if v in cache:
                return cache[v]
            candidate = replace(g, slit_left=v[0], slit_right=v[1],
                                core_depth=v[2], guide_length=v[3])
            try:
                rows, m = evaluate(candidate,args.frequencies,args.angles,spacing)
                if m['central_clear_fraction'] < .75-1e-10:
                    raise ValueError('Clear opening below constraint')
                value = score(rows)
            except ValueError:
                cache[v] = 1e3
                return 1e3
            entry = dict(geometry=asdict(candidate),score=value,rows=rows)
            history.append(entry)
            with (args.out/'history.jsonl').open('a') as f:
                f.write(json.dumps(entry)+'\n')
            cache[v] = value
            if value < best[0]:
                best[:] = [value,candidate]
                (args.out/'best_so_far.json').write_text(json.dumps(entry,indent=2))
            print(f"evaluation {len(history)} score={value:.5f} best={best[0]:.5f}",flush=True)
            return value
        if (args.out/'history.jsonl').exists():
            p.error('Use a fresh output directory for optimisation; history already exists')
        differential_evolution(objective,[(.004,.014),(.004,.014),(.130,.230),(.016,.100)],
                               maxiter=args.iterations,popsize=args.popsize,seed=args.seed,
                               polish=False,workers=1,tol=0,atol=0,
                               x0=[g.slit_left,g.slit_right,g.core_depth,g.guide_length])
        g = best[1]
    final, metrics = evaluate(g,args.frequencies,args.angles,spacing)
    report = dict(model='lossless rigid 2D periodic; screening only; not paper-validated',
                  candidate_status='unverified; run verify_candidate before interpreting performance',
                  geometry=asdict(g), initial_geometry=initial_geometry, mesh=metrics,
                  frequencies=args.frequencies,angles=args.angles,baseline=baseline,
                  final=final,baseline_score=score(baseline),final_score=score(final),
                  unique_evaluations=len(history),seed=args.seed,
                  elapsed_s=time.monotonic()-start,
                  objective='0.7 mean power transmission + 0.3 worst power transmission; equal weights',
                  measured_spectrum_used=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    mask,dx,dz,_ = rasterize(g,spacing)
    result = solve(mask,dx,dz,1000.)
    distances = np.linspace(0,.5,151)
    np.savez_compressed(args.out/'field_1000hz.npz',pressure=result.pressure,solid=mask,
                        dx=dx,dz=dz,frequency=1000.,angle=0.,kx=result.kx,
                        outgoing_modes=result.outgoing_modes,
                        downstream=result.downstream(distances),downstream_z=distances)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(14,5),layout='constrained')
    extent = [0,mask.shape[1]*dx*1000,0,mask.shape[0]*dz*1000]
    axes[0].imshow(mask,origin='lower',extent=extent,cmap='Greys',interpolation='nearest')
    axes[0].set(title='Geometry: rigid solids in black',xlabel='Transverse x (mm)',ylabel='Propagation z (mm)')
    im=axes[1].imshow(abs(result.pressure),origin='lower',extent=extent,cmap='magma',vmin=0,
                      vmax=max(2,float(np.nanpercentile(abs(result.pressure),99))))
    axes[1].set(title=f'1,000 Hz, normal incidence\nT={result.transmission:.3f}, R={result.reflection:.3f}',xlabel='x (mm)')
    fig.colorbar(im,ax=axes[1],label='Pressure / incident pressure')
    for angle in args.angles:
        for label,rows,style in [('Initial',baseline,'--'),('Final',final,'-')]:
            selected=[r for r in rows if r['angle_deg']==angle]
            axes[2].plot([r['frequency_hz'] for r in selected],[r['tl_db'] for r in selected],
                         style+'o',label=f'{label}, {angle:g}°')
    axes[2].set(xlabel='Frequency (Hz)',ylabel='Transmission loss (dB)',title='Lossless 2D screening prediction')
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.2)
    fig.suptitle('Open acoustic wall — research model, not validated outdoor performance')
    fig.savefig(args.out/'summary.png',dpi=160)
    plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
