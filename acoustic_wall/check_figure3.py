"""Fixed Figure 3 dimensions, two independent solvers, no optimisation."""
import json,os
from pathlib import Path
import numpy as np
from .paper_benchmark import CASES,geometry
from .benchmark_mesh import PreparedMesh
from .independent_fem import solve_fem

OUT=Path(__file__).parent/'benchmark'/'figure3_check'

def main():
    OUT.mkdir(parents=True,exist_ok=True);case=CASES[0]
    rows=[]
    for edge in [.00005,.000025,.0000125]:
        for ch in [0,1,2]:
            path=OUT/f'channel{ch}_edge{edge:g}.json'
            if path.exists():rows.append(json.loads(path.read_text()));continue
            s,xe,ze,meta=geometry(case,.0003,'added','centred',isolated=ch,guide=False,corner_step=edge)
            fv=PreparedMesh(s,xe,ze).solve()
            z=(ze[:-1]+ze[1:])/2
            t=fv.forward[0]*np.exp(-1j*fv.q[0]/fv.port_step*(z[-1]-z[0]))
            fem=solve_fem(s,xe,ze);fp=fem.pop('pressure')
            data=dict(channel=ch,slit_mm=[3.25,25,2][ch],edge_m=edge,metadata=meta,
                      fv=dict(amplitude=float(abs(t)),relative_phase_deg=float(np.angle(t,deg=True)),T=fv.transmission,R=fv.reflection,residual=fv.residual,energy_error=fv.balance_error),fem=fem)
            path.write_text(json.dumps(data,indent=2));rows.append(data)
            if edge==.0000125:np.savez_compressed(OUT/f'channel{ch}_field.npz',solid=s,xedges=xe,zedges=ze,fv=fv.pressure,fem=fp)
            print(ch,edge,'FV',data['fv']['amplitude'],data['fv']['relative_phase_deg'],'FEM',fem['amplitude'],fem['relative_phase_deg'],flush=True)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/acoustic-wall-mpl')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    labels=['3.25 mm slit','25 mm blank','2.00 mm slit'];fine=sorted([r for r in rows if r['edge_m']==.0000125],key=lambda r:r['channel'])
    target_amp=case['amplitude'];target_phase=np.angle(np.exp(-1j*(np.array(case['phase_pi'])-1)*np.pi),deg=True)
    for key,marker in [('fv','o'),('fem','x')]:
        axs[0,0].plot(range(3),[r[key]['amplitude'] for r in fine],marker+'-',label=key.upper())
        phase=np.array([r[key]['relative_phase_deg'] for r in fine]);phase=target_phase+np.angle(np.exp(1j*np.deg2rad(phase-target_phase)),deg=True)
        axs[0,1].plot(range(3),phase,marker+'-',label=key.upper())
    axs[0,0].plot(range(3),target_amp,'s--',color='gray',label='Paper, Table 2')
    axs[0,1].plot(range(3),target_phase,'s--',color='gray',label='Paper, conjugated')
    for ax in axs[0]:ax.set_xticks(range(3),labels);ax.legend();ax.grid(alpha=.2)
    axs[0,0].set(title='Complex transmission: amplitude',ylabel='Pressure amplitude / incident amplitude')
    axs[0,1].set(title='Phase relative to the blank channel',ylabel='Degrees (exp(-iωt) convention)')
    for ch,color in [(0,'#187e96'),(2,'#c77d30')]:
        for key,ls in [('fv','-'),('fem','--')]:
            rs=sorted([r for r in rows if r['channel']==ch],key=lambda r:r['edge_m'],reverse=True)
            axs[1,0].plot([r['edge_m']*1e6 for r in rs],[r[key]['amplitude'] for r in rs],'o'+ls,color=color,label=f'{labels[ch]} {key.upper()}')
    axs[1,0].set(title='Two independent discretisations, same rectangles',xlabel='Cell size near edges (µm)',ylabel='Transmission amplitude');axs[1,0].invert_xaxis();axs[1,0].legend(fontsize=8);axs[1,0].grid(alpha=.2)
    a=np.load(OUT/'channel2_field.npz');xe=a['xedges']*1000;ze=a['zedges']*1000
    axs[1,1].pcolormesh(ze,xe,a['solid'].T,cmap='Greys',shading='flat');axs[1,1].set_aspect('equal')
    axs[1,1].set(title='2 mm slit: fixed Figure 3 dimension set',xlabel='Axial position (mm)',ylabel='25 mm clear channel')
    fig.suptitle('Figure 3 check at 2,000 Hz — no geometry optimisation',fontsize=16)
    fig.supxlabel('H = 85.75 mm; 10 ribs × 5 mm; clear width = 25 mm. Equal pitch H/10 is an explicit reconstruction assumption.\nRigid channel walls; lossless 2D; exact rectangle edges. Matching solvers does not establish exact author CAD.',fontsize=10)
    fig.savefig(OUT/'comparison.png',dpi=175);plt.close(fig)
    summary=[]
    for r in fine:
        ch=r['channel'];summary.append(dict(slit_mm=r['slit_mm'],paper_amplitude=target_amp[ch],paper_relative_phase_deg=float(target_phase[ch]),fv=r['fv'],fem=r['fem'],amplitude_method_difference=abs(r['fv']['amplitude']-r['fem']['amplitude']),phase_method_difference_deg=float(abs(np.angle(np.exp(1j*np.deg2rad(r['fv']['relative_phase_deg']-r['fem']['relative_phase_deg'])),deg=True)))))
    (OUT/'report.json').write_text(json.dumps(dict(geometry_exactness='Published dimensions exact; axial rib placement inferred, not verified author CAD.',results=summary,mesh_runs=rows),indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
