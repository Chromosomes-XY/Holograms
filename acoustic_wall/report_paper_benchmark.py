"""Render benchmark evidence and record failures without tuning reference data."""
import csv,json,os
from pathlib import Path
import numpy as np
from .paper_benchmark import ROOT,CASES


def main():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    all_rows=[]
    for p in (ROOT/'results').glob('*_f*.json'):
        d=json.loads(p.read_text());d['file']=p.name;all_rows.append(d)
    missing=[]
    for case in CASES[1:]:
        for h,corner in [(.0004,None),(.0002,None),(.0004,.000025)]:
            have={d['frequency_hz'] for d in all_rows if d['case']['label']==case['label'] and d['width_convention']=='inclusive' and d['rib_layout']=='centred' and d['boundary']=='rigid' and d['step_m']==h and d.get('corner_step_m')==corner}
            absent=set(np.arange(1700,2301,20))-have
            if absent:missing.append((case['label'],h,corner,len(absent)))
    if missing:raise RuntimeError(f'Benchmark sweeps still incomplete (case, step, edge step, missing count): {missing}')
    refs=list(csv.DictReader((ROOT/'reference_fig5a.csv').open()))
    ref=np.array([[float(r[k]) for k in ['openness','frequency_hz','T']] for r in refs])
    fig,axes=plt.subplots(6,2,figsize=(15,14),layout='constrained',gridspec_kw={'width_ratios':[1,1.35]})
    summary=[];fieldmax=0
    for i,case in enumerate(CASES[1:]):
        label=case['label'];ax=axes[i,0];f_ax=axes[i,1]
        reference=ref[ref[:,0]==label];ax.scatter(reference[:,1],reference[:,2],s=2,color='#626773',alpha=.7,label='Paper (digitised)')
        rows=[d for d in all_rows if d['case']['label']==label and d['width_convention']=='inclusive' and d['rib_layout']=='centred' and d['boundary']=='rigid']
        curves=[]
        for step,corner,color,ls,name in [(.0004,None,'#9faab6',':','0.4 mm'),(.0002,None,'#d29349','--','0.2 mm'),(.0004,.000025,'#167d9a','-','25 µm edge cells')]:
            rs=sorted([d for d in rows if abs(d['step_m']-step)<1e-9 and d.get('corner_step_m')==corner],key=lambda d:d['frequency_hz'])
            if len(rs)>2:
                ax.plot([d['frequency_hz'] for d in rs],[d['T'] for d in rs],ls,color=color,lw=1.5,label=name)
                curves.append(rs)
        # Use the finest local-edge field if available; no field rescaling.
        targets=[d for d in rows if d['frequency_hz']==2000]
        target=min(targets,key=lambda d:d.get('corner_step_m') or d['step_m'])
        fname=target['file'].replace('_f2000.json','_field.npz');a=np.load(ROOT/'results'/fname)
        xe=a['xedges']*1000;ze=a['zedges']*1000;distance=a['distance']*1000
        before=a['before'];after=a['after'];p=a['pressure'];solid=a['solid'];W=target['width_m']*1000;L=target['length_m']*1000
        fieldmax=max(fieldmax,float(np.nanmax(abs(p))))
        zc=(ze[:-1]+ze[1:])/2;xc=(xe[:-1]+xe[1:])/2
        cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('#262b31')
        im=f_ax.pcolormesh(ze,xe,np.ma.array(p.real.T,mask=solid.T),cmap=cm,vmin=-2,vmax=2,shading='flat',rasterized=True)
        f_ax.pcolormesh(zc[0]-distance[::-1],xc,before[::-1].real.T,cmap=cm,vmin=-2,vmax=2,shading='nearest',rasterized=True)
        f_ax.pcolormesh(zc[-1]+distance,xc,after.real.T,cmap=cm,vmin=-2,vmax=2,shading='nearest',rasterized=True)
        f_ax.set(xlim=(-170,L+170),ylim=(0,W),ylabel='Across cell (mm)',title=f'2,000 Hz: computed T = {target["T"]:.3g}')
        f_ax.set_aspect('equal',adjustable='box')
        for z in [0,L]:f_ax.axvline(z,color='#444',lw=.6,ls=':')
        ax.set(xlim=(1700,2300),ylim=(-.015,1.02),ylabel='Power transmission T',title=f'Nominal openness {label:.2f}')
        ax.axhline(.1,color='#c34747',ls='--',lw=.9);ax.axvline(2000,color='#888',ls=':',lw=.8);ax.grid(alpha=.12)
        if i==0:ax.legend(fontsize=8,loc='upper right')
        if i==5:ax.set_xlabel('Frequency (Hz)');f_ax.set_xlabel('Propagation distance (mm); module starts at 0')
        best_curve=curves[-1] if curves else []
        if len(best_curve)>=30:
            fs=np.array([d['frequency_hz'] for d in best_curve]);ts=np.array([d['T'] for d in best_curve])
            # Compare at sampled simulation frequencies with observed raster points within 3 Hz.
            pairs=[]
            for f,t in zip(fs,ts):
                near=reference[abs(reference[:,1]-f)<=3,2]
                if len(near):pairs.append([f,t,float(np.median(near))])
            pairs=np.array(pairs)
            rmse=float(np.sqrt(np.mean((pairs[:,1]-pairs[:,2])**2)))
        else:rmse=None;pairs=np.empty((0,3))
        row=dict(openness=label,T_2000=target['T'],TL_2000_db=float(-10*np.log10(max(target['T'],1e-16))),
                 final_target_file=target['file'],spectrum_rmse=rmse,compared_frequency_count=len(pairs),
                 curve_file_count=len(best_curve),paper_target_claim_T_below=.1,
                 target_claim_reproduced=target['T']<.1,central_clear_fraction=target['central_clear_fraction'])
        summary.append(row)
    fig.suptitle('Rectangular PGUOM benchmark — published dimensions, no transmission-curve fitting\nLeft: reference versus local solver     Right: local pressure fields at 2,000 Hz',fontsize=15)
    fig.supxlabel('Rigid transverse boundaries; wall-inclusive reconstruction; lossless 2D. Curves sampled every 20 Hz.\nPressure colours fixed at ±2 × incident pressure (clipped); fields are local simulations, not paper images.',fontsize=10)
    fig.colorbar(im,ax=axes[:,1],fraction=.015,pad=.015,label='Re(p) / incident pressure',extend='both')
    fig.savefig(ROOT/'benchmark_comparison.png',dpi=165);fig.savefig(ROOT/'benchmark_comparison.svg');plt.close(fig)

    # Independent checks: ports, single-channel phase, mesh sensitivity, boundary sensitivity.
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    ports=json.loads((ROOT/'results'/'port_benchmark.json').read_text())['rows'];O=[r['openness'] for r in ports]
    axs[0,0].plot(O,[r['phase_only_T'] for r in ports],'o-',label='Equal incoming amplitudes')
    axs[0,0].plot(O,[r['paper_amplitude_T'] for r in ports],'o-',label='Table 1 incoming amplitudes')
    axs[0,0].set(title='Independent-port reconstruction',xlabel='Nominal openness',ylabel='Outgoing / total incoming power');axs[0,0].legend(fontsize=8)
    audit_path=ROOT/'results'/'isolated_audit_inclusive_centred_refined.json'
    audit=json.loads(audit_path.read_text())
    phase_errors=[];amplitude_errors=[]
    for row in audit:
        c=row['case'];v=row['response'];phase=np.array([v[j]['relative_phase_deg'] for j in [0,2]])
        # COMSOL uses +i wt; this solver uses -i wt. Conjugate once, consistently.
        desired=-(np.array(c['phase_pi'])[[0,2]]-c['phase_pi'][1])*180
        err=np.angle(np.exp(1j*np.deg2rad(phase-desired)),deg=True)
        phase_errors.append(err);amplitude_errors.append(np.array([v[j]['amplitude'] for j in [0,2]])-np.array(c['amplitude'])[[0,2]])
    phase_errors=np.array(phase_errors)
    for j,name in enumerate(['Side 1','Side 3']):axs[0,1].plot([c['label'] for c in CASES],phase_errors[:,j],'o-',label=name)
    axs[0,1].axhline(0,color='gray',lw=.7);axs[0,1].axhspan(-10,10,color='#b5d5bc',alpha=.3)
    axs[0,1].set(title='Isolated channels versus Table 2 (12.5 µm edges)',xlabel='Nominal openness',ylabel='Relative-phase error (degrees)');axs[0,1].legend(fontsize=8)
    meshes=[(.0004,None,'0.4'),(.0002,None,'0.2'),(.0001,None,'0.1'),(.0004,.00005,'edge 50µm'),(.0004,.000025,'edge 25µm'),(.0002,.0000125,'edge 12.5µm')]
    convergence=[]
    for c in CASES[1:]:
        vals=[]
        for h,edge,name in meshes:
            match=[d for d in all_rows if d['case']['label']==c['label'] and d['frequency_hz']==2000 and d['width_convention']=='inclusive' and d['rib_layout']=='centred' and d['boundary']=='rigid' and abs(d['step_m']-h)<1e-10 and d.get('corner_step_m')==edge]
            vals.append(match[0]['T'] if match else np.nan)
        axs[1,0].plot(range(len(meshes)),vals,'o-',label=f'{c["label"]:.2f}')
        convergence.append(dict(openness=c['label'],T=vals,last_change=abs(vals[-1]-vals[-2]) if np.isfinite(vals[-1]) else None))
    axs[1,0].set(xticks=range(len(meshes)),xticklabels=[m[2] for m in meshes],title='Same geometry: mesh refinement at 2,000 Hz',ylabel='Power transmission T');axs[1,0].tick_params(axis='x',rotation=25);axs[1,0].legend(ncol=3,fontsize=8)
    for boundary,marker in [('rigid','o'),('periodic','s')]:
        ds=sorted([d for d in all_rows if d['frequency_hz']==2000 and d['boundary']==boundary and d['step_m']==.0004 and d.get('corner_step_m')==.000025 and d['width_convention']=='inclusive' and d['rib_layout']=='centred' and d['case']['label']>.34],key=lambda d:d['case']['label'])
        axs[1,1].plot([d['case']['label'] for d in ds],[d['T'] for d in ds],marker+'-',label=boundary)
    axs[1,1].set(title='Boundary sensitivity (25 µm edge cells)',xlabel='Nominal openness',ylabel='Power transmission T');axs[1,1].legend(fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.15)
    fig.suptitle('Benchmark diagnostics — numerical verification and paper agreement are separate checks',fontsize=14)
    fig.savefig(ROOT/'benchmark_diagnostics.png',dpi=165);plt.close(fig)
    extra_refinement=[dict(openness=d['case']['label'],corner_step_m=d.get('corner_step_m'),T=d['T']) for d in all_rows if d.get('corner_step_m')==.00000625 and d['frequency_hz']==2000]
    report=dict(extra_refinement=extra_refinement,status='Paper reproduction NOT established',cases=summary,convergence=convergence,
                isolated_phase_error_deg=phase_errors.tolist(),isolated_amplitude_error=np.array(amplitude_errors).tolist(),
                max_energy_error=max(d['energy_error'] for d in all_rows),max_linear_residual=max(d['residual'] for d in all_rows),
                solved_physical_cases=len(all_rows),maximum_field_amplitude=fieldmax,
                reference_uncertainty_T=.02,reference_uncertainty_hz=3,
                limitations=['No author CAD or COMSOL files supplied.','Width and axial end-placement conventions audited, not fitted to Figure 5.',
                 'Figure 5 lateral boundary is not unambiguously specified by its caption; rigid is the primary reconstruction, periodic is a sensitivity check.',
                 'No thermoviscosity or 3D extrusion; no fabrication validation.','20 Hz sweep can miss narrow peaks.',
                 'Independent-port model is a documented reconstruction, not an exact author-model reproduction.'])
    (ROOT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    table='\n'.join(f'| {r["openness"]:.2f} | {r["T_2000"]:.5g} | {r["TL_2000_db"]:.2f} | {r["spectrum_rmse"]:.3f} | {r["compared_frequency_count"]} |' for r in summary)
    text=f'''# Rectangular PGUOM benchmark

**Outcome: the paper's physical Figure 5 results are not yet reproduced.**
The numerical solver passes conservation and independent assembly checks, but
that is not evidence that this reconstructed geometry is the authors' model.
No geometry was fitted to the Figure 5 curves.

## Comparison

![Reference comparison]({ROOT/'benchmark_comparison.png'})

| Nominal openness | Local T at 2000 Hz | Local loss (dB) | Spectrum T RMSE | Compared frequencies |
|---|---:|---:|---:|---:|
{table}

The paper reports T < 0.1 at 2000 Hz for all six cases. Reference spectra are
colour-digitised from the supplied Figure 5a screenshot, not author raw data.
Approximate reading uncertainty is 0.02 in T and 3 Hz away from steep features.
Overlapped/obscured pixels are omitted. RMSE compares observed reference points
within 3 Hz of local sampled frequencies; it is not a fitted-curve score.
See `reference_digitisation_check.png` and `reference_provenance.json`.

## What is benchmarked

1. Exact agreement of the new nonuniform finite-volume assembly with the original
   solver on the same uniform grid. Empty and closed guides; reciprocity; scaling;
   conservation; independently driven inlet ports; mirror symmetry from S1.
2. The paper's Table 1 phase prescription and amplitudes in three separately
   excited incoming waveguide ports. Zero-thickness rigid separators end at a
   common guide; 40 mm isolated entrance and 110 mm common downstream region,
   total transverse width 75 mm. Incoming power is summed across all three
   ports. This is an explicitly specified reconstruction, not the authors'
   unprovided COMSOL port setup. It confirms the expected amplitude-tuning trend.
3. Table 2 single-channel complex transmission, referenced to an empty channel
   between the same planes. COMSOL's +iωt phasors are conjugated for our -iωt
   convention. Table 2 channel numbering mirrors the Figure 3 example; its order
   is preserved. No per-case sign or phase adjustment is made.
   Figure 5 labels its pressure scale only as max/min; pressure-colour agreement
   is therefore qualitative. Our fields use one fixed incident-normalised scale.
4. Six fixed physical geometries with published slit widths and waveguide lengths,
   1700–2300 Hz in 20 Hz steps, and 2000 Hz pressure fields. Hard side walls are
   the primary reconstruction. Periodic boundaries, alternate wall-width
   accounting and rib end placement are separate sensitivity checks.

![Diagnostics]({ROOT/'benchmark_diagnostics.png'})

## Geometry and mesh assumptions

Published H=85.75 mm, n=10 rib pairs, axial rib width h=5 mm, longitudinal
wall thickness tt=2 mm. Each side uses the Table 2 slit width. Per-side guide
lengths are 0.3H, 0.2H, 0.1H, 0.5H, 0.5H and 0.5H (Figure 5 caption).
The main reconstruction places equally spaced rib centres at (j+1/2)H/n.
The axial end setback is inferred, not separately dimensioned by the source.

Main width convention: Wi spans wall centrelines; clear air width is Wi−2 mm.
Thus nominal openness is not the actual clear fraction: the 0.73 case has
53/75 = 70.67% central clearance in this reconstruction. The alternative treats
Wi as clear air and adds walls, giving an 81 mm repeat. Both conventions are
recorded; neither is asserted to be verified CAD. Do not use nominal openness
as a finished-wall transparency measurement.

Mesh lines coincide with all geometry edges. The 0.58 and 0.53 mm slits remain
distinct at every mesh level. Maximum-cell meshes of 0.4, 0.2 and 0.1 mm are
checked, plus grids with 50, 25 and 12.5 micrometre cells near rectangle edges.
The 0.73 case additionally uses 6.25 micrometre edge cells.
Mesh steps here are local acoustic discretisation, not manufacturing tolerances.
All propagating modal power is counted, including higher duct modes above cutoff.
Outgoing modal boundaries replace PMLs for these homogeneous rigid/periodic guides.

Maximum |R+T−1| across saved physical solves: {report['max_energy_error']:.3g}.
Maximum relative linear-system residual: {report['max_linear_residual']:.3g}.
Neither metric measures geometry fidelity or continuum mesh convergence.

## Consequence for the garden-wall design

The previous approximately 25 dB prediction belongs to an unvalidated 2D
candidate. This benchmark does not validate that number or establish the paper's
mechanism in that candidate. First resolve the Table 2 complex-transmission
mismatch and geometry/port definitions; then require agreement across the six
spectra and mesh levels before optimising a 75%-clear 1000 Hz wall.

Source files and provenance: `sources/paper.pdf`, `sources/supplement.pdf`.
Paper: https://doi.org/10.1038/s41598-025-04885-6
COMSOL phasor convention:
https://doc.comsol.com/6.3/doc/com.comsol.help.aco/aco_ug_pressure.05.002.html

See `README.md` for commands. All raw solve metrics and saved 2000 Hz fields
are in `results/`; the benchmark summary is `report.json`.
'''
    (ROOT/'BENCHMARK.md').write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
