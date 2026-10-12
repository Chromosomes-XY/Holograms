"""Render the measured solver disagreement, without presenting it as validated performance."""
import os,json
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
OUT=Path(__file__).parent/'results'/'figure3_scaled_1000'
a=np.load(OUT/'diagnostic_fields.npz');x=a['xedges'];z=a['zedges'];s=a['solid'];fv=a['fv'];fp=a['fem']
aligned=fp*np.exp(-1j*2*np.pi*1000/343*(z[1]-z[0])/2)
fem=(aligned[:-1,:-1]+aligned[1:,:-1]+aligned[:-1,1:]+aligned[1:,1:])/4
rows=json.loads((OUT/'diagnostic_runs.json').read_text())
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
axs[0].pcolormesh(z*1000,x*1000,s.T,cmap=ListedColormap(['#eff3f6','#32363d']),shading='flat')
axs[0].set_title('Fixed geometry, dimensions ×2')
for ax,field,title in zip(axs[1:],[fv,fem],['Finite volume: T = 0.0420','Independent FEM: T = 0.5613']):
    im=ax.pcolormesh(z*1000,x*1000,np.ma.masked_where(s,field.real).T,cmap='RdBu_r',vmin=-2,vmax=2,shading='flat')
    ax.set_title(title)
for ax in axs:
    ax.set_facecolor('#32363d');ax.set_aspect('equal');ax.set_xlabel('Axial z (mm) →');ax.set_ylabel('Transverse x (mm)')
fig.colorbar(im,ax=axs[1:],label='Re(p) / incident pressure; colours clipped at ±2',shrink=.65)
fig.suptitle('1,000 Hz pressure solution — PROVISIONAL: solvers do not agree',fontsize=16)
fig.supxlabel('Same 1.2 mm maximum-step / 0.1 mm edge mesh · incident phases aligned\n171.5 mm core + 68.6 mm guide at each end · 162 mm width · 50 mm clear centre (30.9%)\nRigid lossless 2D duct; inferred rib spacing. These fields are not a validated wall-performance prediction.',fontsize=11)
fig.savefig(OUT/'pressure_1000_diagnostic.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for key,label in [('fv_T','Finite volume'),('fem_T','Triangular FEM')]:ax.plot(range(3),[r[key] for r in rows],'o-',label=label)
ax.set_xticks(range(3),['4.0 / 0.4','1.2 / 0.1','0.6 / 0.1']);ax.set(xlabel='Maximum / edge cell size (mm)',ylabel='Transmitted power fraction T',ylim=(0,1.05),title='Refinement check fails: no reliable attenuation estimate');ax.legend();ax.grid(alpha=.25)
fig.savefig(OUT/'mesh_disagreement.png',dpi=160)
report=dict(status='UNVALIDATED: mesh and method disagreement',frequency_hz=1000,geometry=dict(core_mm=171.5,guide_each_mm=68.6,total_depth_mm=308.7,computational_width_mm=162,clear_channel_widths_mm=[50,50,50],slits_mm=[6.5,4],rib_width_mm=10,wall_mm=4,central_clear_fraction=50/162),runs=rows,plotted_run=rows[1],max_pressure_amplitude=dict(fv=float(np.nanmax(abs(fv))),fem=float(np.nanmax(abs(fp)))),limitations=['Rigid lossless 2D duct, normal incidence','Published dimensions scaled by two; rib placement inferred','No optimisation; does not meet 75% clear-view target','No validated attenuation figure from these runs'])
(OUT/'report.json').write_text(json.dumps(report,indent=2))
(OUT/'README.md').write_text('''# 1,000 Hz fixed-geometry solve — unvalidated

The latest Figure 3 reconstruction was scaled by two. The two solvers disagree and refinement does not establish convergence. Do not use either low-transmission result as a performance prediction.

Dimensions: core 171.5 mm, guides 68.6 mm each, total depth 308.7 mm, computational transverse width 162 mm, three clear channels 50 mm each, side slits 6.5 and 4 mm, axial ribs 10 mm, walls 4 mm. Equal rib pitch remains inferred. Central clear-view fraction including walls is 30.86%, not 75%.

See diagnostic_runs.json for all completed full-device runs; diagnostic_fields.npz contains the middle-grid FV cell and FEM nodal complex pressures and exact mesh/mask. pressure_1000_diagnostic.png displays both on one pressure colour scale with FEM incident phase aligned to the FV port centre. The model has rigid transverse boundaries and exact modal radiation at the axial ends; no absorption or thermoviscous losses.

Re-render: `.venv/bin/python -m acoustic_wall.plot_scaled_1000_diagnostic`.
The full fine-grid driver is solve_scaled_1000.py; its longer refinement run was stopped when disagreement appeared. Tests of independent_fem passed (three tests). scaling_check.json separately compares the identical dimensionless coarse mesh at 2,000 Hz and at 1,000 Hz after scaling by two. Passing scaling does not resolve the full-device solver discrepancy.
''')
print(report)
