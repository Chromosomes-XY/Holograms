"""Summarise coarse-grid and PML sensitivity for the finite-array experiment."""
import json
import numpy as np
from .finite_array_2d import OUT,FiniteArray

def average(q,x,z):
    area=np.gradient(z)[:,None]*np.gradient(x)[None,:]
    mask=(abs(x)[None,:]<4)&(z[:,None]>.7)&(z[:,None]<3.5)
    a=float(np.sum(abs(q['incident'][mask])**2*area[mask]))
    b=float(np.sum(abs(q['p'][mask])**2*area[mask]))
    return float(10*np.log10(a/b))

def main():
    path=OUT/'pml_check_1000.npz'
    if not path.exists():
        m=FiniteArray(pml=1.5);p,_,r=m.solve(1000);b,_,br=m.solve(1000,empty=True)
        np.savez_compressed(path,x=m.x,z=m.z,p=p,incident=b,residual=r,baseline_residual=br)
    mesh=np.load(OUT/'mesh.npz');coarse=np.load(OUT/'field_1000.npz');fine=np.load(OUT/'fine_1000.npz');pml=np.load(OUT/'pml_check_1000.npz')
    checks=dict(region='Area-integrated mean-square pressure over x=-4..4 m, z=0.7..3.5 m; not transmitted flux',
                coarse_1000_db=average(coarse,mesh['x'],mesh['z']),fine_1000_db=average(fine,fine['x'],fine['z']),
                thicker_pml_1000_db=average(pml,pml['x'],pml['z']))
    q=np.load(OUT/'broadband.npz');area=np.gradient(mesh['z'])[:,None]*np.gradient(mesh['x'])[None,:]
    mask=(abs(mesh['x'])[None,:]<4)&(mesh['z'][:,None]>.7)&(mesh['z'][:,None]<3.5)
    checks['approximate_whole_spectrum_region_db']=float(10*np.log10(np.sum((q['baseline']+q['tail'])[mask]*area[mask])/np.sum((q['weighted']+q['tail'])[mask]*area[mask])))
    checks['note']='Fine/coarse differences are numerical sensitivity, not an error bound. Frequency bands are represented at their centres only.'
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
