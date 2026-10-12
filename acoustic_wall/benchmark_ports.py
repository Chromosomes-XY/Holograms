"""Three independently driven incoming waveguide ports, not imposed exit pressure.

Rigid zero-thickness separators end at a common outlet. Report outgoing power
normalised by the sum of incoming port powers. This reconstructs the described
port concept; author COMSOL files / port normalisation were not supplied.
"""
import json,os
import numpy as np
from .benchmark_mesh import PreparedMesh,segmented_grid
from .paper_benchmark import ROOT


def make_ports(openness,step=.0005):
    W=.075;side=(1-openness)*W/2
    xe=segmented_grid([0,side,W-side,W],step)
    # Equal axial cells so uniform analytic phase is exact up to FD dispersion.
    ze=np.arange(0,.15+step/2,step)
    groups=[(0,np.argmin(abs(xe-side))),(np.argmin(abs(xe-side)),np.argmin(abs(xe-(W-side)))),(np.argmin(abs(xe-(W-side))),len(xe)-1)]
    blocked=np.zeros((len(ze)-1,len(xe)-2),bool)
    for _,b in groups[:-1]:blocked[(ze[:-1]+ze[1:])/2<.04,b-1]=True
    return PreparedMesh(np.zeros((len(ze)-1,len(xe)-1),bool),xe,ze,'rigid',blocked,groups)


def run():
    from scipy.optimize import minimize_scalar
    rows=[]
    for i,side in enumerate(np.arange(25,4.9,-2.5)):
        O=(75-2*side)/75;model=make_ports(O)
        # Conjugate the paper/COMSOL +iwt phase convention for this -iwt solver.
        phase=np.array([1,0,-1])*(1+O)*np.pi/2
        # Solve each inlet once and use linearity to evaluate continuous amplitudes.
        basis=[]
        for j in range(3):
            a=np.zeros(3,complex);a[j]=np.exp(1j*phase[j]);basis.append(model.solve(incident=a))
        flux=np.where(abs(basis[0].q.imag)<1e-9,np.sin(basis[0].q.real)/model.h,0)
        widths=np.array([side,75-2*side,side])/1000
        coeff=np.array([r.forward for r in basis])
        def T(a):
            amps=np.array([1,a,1]);v=amps@coeff
            return float(np.sum(flux*abs(v)**2)*.075/(flux[0]*np.sum(widths*amps**2)))
        opt=minimize_scalar(T,bounds=(.1,1),method='bounded');paper_a=1-.1*i
        validation=model.solve(incident=np.array([1,paper_a,1])*np.exp(1j*phase))
        if abs(validation.transmission-T(paper_a))>1e-8 or validation.balance_error>1e-8:raise RuntimeError('Port power verification failed')
        row=dict(openness=O,paper_centre_amplitude=paper_a,phase_only_T=T(1),paper_amplitude_T=T(paper_a),
                 local_optimum_centre=float(opt.x),local_optimum_T=float(opt.fun),energy_error=validation.balance_error)
        rows.append(row);print(row,flush=True)
    (ROOT/'results'/'port_benchmark.json').write_text(json.dumps(dict(rows=rows,normalisation='sum of incoming port powers',
      limitations='Independent incoming fundamental modes with rigid separators. Not an exact replication of unpublished COMSOL port setup.'),indent=2))

if __name__=='__main__':run()
