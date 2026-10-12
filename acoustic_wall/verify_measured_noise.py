"""Additional spatial and directional checks of the selected noise design."""
from .optimise_measured_noise import OUT, integrate, geometry, PreparedMesh, EDGES, FC, WEIGHT
import json
import numpy as np

def spatial(v):
    # Identical quadrature on both meshes isolates spatial discretisation error.
    nodes,weights=np.polynomial.legendre.leggauss(5)
    rows=[]
    for h in [.0015,.00075]:
        s,x,z=geometry(v,h)
        mesh=PreparedMesh(s,x,z,boundary='periodic')
        for f in [800,1000,1250]:
            i=int(np.flatnonzero(FC==f)[0]);lo,hi=EDGES[i:i+2]
            fs=(lo+hi)/2+(hi-lo)/2*nodes
            results=[mesh.solve(float(q)) for q in fs]
            rows.append(dict(h=h,band=f,T=float(weights@np.array([r.transmission for r in results])/2),
                             maximum_energy_error=max(r.balance_error for r in results)))
        print('Spatial check complete',h,flush=True)
    (OUT/'spatial_checks.json').write_text(json.dumps(dict(v=v,rows=rows),indent=2))
    return rows

def main():
    report=json.loads((OUT/'report.json').read_text())
    v=report['selected']['v']
    path=OUT/'spatial_checks.json'
    cached=json.loads(path.read_text()) if path.exists() else {}
    rows=cached['rows'] if cached.get('v')==v else spatial(v)
    angles=[]
    for angle in [0,-30,30,-60,60]:
        result=integrate(v,5,'search',angle=angle)
        angles.append(result)
        print('Angle',angle,'remaining weighted power',result['J'],flush=True)
    result=dict(spatial_checks=rows,angle_checks=angles,
                note='Angles are separate full-profile screening estimates, using five quadrature points and the search mesh; they are not part of the optimisation objective. Spatial checks use identical five-point quadrature at both resolutions.')
    (OUT/'additional_checks.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
