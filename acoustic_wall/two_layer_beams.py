"""Two layers with independent angles, dimensions, lateral placement and gap."""
import json
import numpy as np
from .angled_rectangles import OUT,evaluate,geometry

def design(v):
    L1,t1,B1,s1,L2,t2,B2,s2,gap,swap=map(int,v)
    assert B1+B2>=75 and gap%5==0 and 10<=gap<=100 and s1 in [-1,1] and s2 in [-1,1] and swap in [0,1]
    beams=[]
    for i,(L,t,B,sign) in enumerate([(L1,t1,B1,s1),(L2,t2,B2,s2)]):
        assert L%5==0 and t%5==0 and B in [15,25,35,45,55,65,75] and 80<=L<=250 and 10<=t<=30 and t<=B
        angle=np.arcsin(B/np.hypot(L,t))-np.arctan2(t,L)
        span=L*np.cos(angle)+t*np.sin(angle)
        left=(i==0) != bool(swap);xc=112.5+B/2 if left else 187.5-B/2
        zc=5*np.ceil((span/2+10)/5) if i==0 else 5*np.ceil((beams[0]['z_mm']+beams[0]['axial_span_mm']/2+gap+span/2)/5)
        beams.append(dict(length_mm=L,thickness_mm=t,projected_width_mm=B,angle_deg=float(sign*np.rad2deg(angle)),x_mm=float(xc),z_mm=float(zc),axial_span_mm=float(span)))
    depth=float(5*np.ceil((beams[-1]['z_mm']+beams[-1]['axial_span_mm']/2+10)/5));assert depth<=600
    intervals=sorted([(b['x_mm']-b['projected_width_mm']/2,b['x_mm']+b['projected_width_mm']/2) for b in beams])
    assert intervals[0][1]>=intervals[1][0] and abs(intervals[-1][1]-intervals[0][0]-75)<1e-10
    assert all(b['x_mm']%5==0 and b['z_mm']%5==0 for b in beams)
    return dict(beams=beams,depth_mm=depth,rows=2,clear_fraction=.75,projected_blockage_mm=75,minimum_gap_mm=gap,parameters=list(map(int,v)))

SEEDS=[[150,15,75,1,150,15,75,-1,20,0],[150,15,55,1,150,15,55,-1,20,0],
       [100,15,55,1,200,15,55,-1,20,0],[150,15,45,1,150,15,65,-1,30,0],
       [150,15,55,1,150,15,55,1,50,0],[200,15,75,1,100,15,25,-1,30,1]]

def mutate(v,rng):
    for _ in range(1000):
        q=np.array(v);indices=rng.choice(10,int(rng.integers(1,5)),replace=False)
        for i in indices:
            if i in [0,4]:q[i]=np.clip(q[i]+rng.choice([-25,-15,-10,10,15,25]),80,250)
            elif i in [1,5]:q[i]=np.clip(q[i]+rng.choice([-5,5]),10,30)
            elif i in [2,6]:q[i]=np.clip(q[i]+rng.choice([-20,-10,10,20]),15,75)
            elif i in [3,7]:q[i]*=-1
            elif i==8:q[i]=np.clip(q[i]+rng.choice([-20,-10,-5,5,10,20]),10,100)
            else:q[i]=1-q[i]
        try:design(q)
        except AssertionError:continue
        if q.tolist()!=list(v):return q.tolist()
    raise RuntimeError('No valid mutation')

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True);path=OUT/'two_layer_history.json';history=json.loads(path.read_text()) if path.exists() else []
    rng=np.random.default_rng(218)
    for i in range(len(history),42):
        if i<len(SEEDS):v=SEEDS[i]
        else:
            leaders=sorted(history,key=lambda q:q['J'])[:5]
            parent=leaders[0] if i%3 else leaders[int(rng.integers(len(leaders)))];v=mutate(parent['design']['parameters'],rng)
        r=evaluate(design(v),n=3,h=.0025);history.append(r);path.write_text(json.dumps(history))
        if r['J']<=min(q['J'] for q in history):print('Two-layer best',i+1,r['J'],r['design'],flush=True)
        elif (i+1)%5==0:print('Two-layer progress',i+1,flush=True)
    unique=[]
    for q in sorted(history,key=lambda q:q['J']):
        if q['design'] not in unique:unique.append(q['design'])
        if len(unique)==3:break
    finalists=[]
    for d in unique:
        q=evaluate(d,n=9,h=.0025,dominant_n=33);finalists.append(q);print('Two-layer finalist',q['J'],flush=True)
    (OUT/'two_layer_report.json').write_text(json.dumps(dict(finalists=finalists,best=min(finalists,key=lambda q:q['J']),proposals=len(history),
        scope='Independent stock lengths/thicknesses, signed angles, lateral offsets and axial spacing. Combined projected shadow fixed to 75 mm in 300 mm pitch. Angles and offsets must obey that shared constraint.'),indent=2))

if __name__=='__main__':main()
