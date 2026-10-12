"""Use a finer explicit mesh for the five dominant bands in the finite survey."""
import json,time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .finite_array_2d import FiniteArray,OUT,FC,P,G,plot

def main():
    fine=FiniteArray(h=.01,far=.02)
    for f in [630,800,1000,1250,1600]:
        path=OUT/f'fine_{f}.npz'
        if path.exists():continue
        start=time.time();p,_,r=fine.solve(f);b,_,br=fine.solve(f,empty=True)
        np.savez_compressed(path,x=fine.x,z=fine.z,p=p,incident=b,residual=r,baseline_residual=br)
        print('Refined',f,'seconds',round(time.time()-start,1),flush=True)
    coarse=FiniteArray();weighted=np.zeros(coarse.solid.shape);baseline=np.zeros_like(weighted);tail=np.zeros_like(weighted)
    coordinates=np.stack([coarse.Z.ravel(),coarse.X.ravel()],axis=-1)
    for f,power,gain in zip(FC,P,G):
        if power==0:continue
        if f<=1600:
            path=OUT/f'fine_{f:g}.npz';refined=path.exists();q=np.load(path if refined else OUT/f'field_{f:g}.npz')
            def values(key):
                a=abs(q[key])**2
                if refined:a=RegularGridInterpolator((q['z'],q['x']),a,method='nearest')(coordinates).reshape(coarse.solid.shape)
                return a
            weighted+=power*gain*values('p');baseline+=power*gain*values('incident')
        else:tail+=power*gain*abs(coarse.incident(f))**2
    np.savez_compressed(OUT/'broadband.npz',weighted=weighted,baseline=baseline,tail=tail)
    report=json.loads((OUT/'report.json').read_text())
    report['refined_bands']=[630,800,1000,1250,1600]
    report['refined_mesh']=dict(wall_step_m=.01,away_step_m=.02,air_unknowns=int(fine.n))
    report['refined_weight_fraction']=float((P*G)[np.isin(FC,report['refined_bands'])].sum()/(P*G).sum())
    report['limitations'].append('Initial coarse 1 kHz result differed by 5.6 dB from refinement in receiver-region average; final dominant bands use refined mesh, but no claim of spatial convergence')
    (OUT/'report.json').write_text(json.dumps(report,indent=2));plot(coarse,weighted,baseline,tail,refined=True)
    print('Maps regenerated with dominant bands refined',flush=True)

if __name__=='__main__':main()
