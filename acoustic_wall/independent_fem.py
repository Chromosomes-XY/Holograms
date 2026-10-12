"""Independent linear-triangle FEM check of rigid 2D pressure acoustics.

Consistent triangle mass, natural rigid boundaries, and multimode radiation
operators from the 1D boundary finite-element eigenproblem. exp(-i omega t).
This is a cross-check, not COMSOL or an author model.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def solve_fem(solid,xe,ze,frequency=2000.,c=343.,from_right=False):
    nz,nx=solid.shape
    zn,xn=np.meshgrid(ze,xe,indexing='ij');coords=np.column_stack([xn.ravel(),zn.ravel()])
    j,i=np.where(~solid);v=j*(nx+1)+i
    triangles=np.concatenate([np.column_stack([v,v+1,v+nx+2]),np.column_stack([v,v+nx+2,v+nx+1])])
    active=np.unique(triangles);ids=np.full(len(coords),-1,int);ids[active]=np.arange(len(active))
    coords=coords[active];triangles=ids[triangles];p=coords[triangles]
    area=((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))/2
    b=np.column_stack([p[:,1,1]-p[:,2,1],p[:,2,1]-p[:,0,1],p[:,0,1]-p[:,1,1]])
    d=np.column_stack([p[:,2,0]-p[:,1,0],p[:,0,0]-p[:,2,0],p[:,1,0]-p[:,0,0]])
    ke=(b[:,:,None]*b[:,None,:]+d[:,:,None]*d[:,None,:])/(4*area[:,None,None])
    me=area[:,None,None]*(np.ones((3,3))+np.eye(3))[None,:,:]/12
    rows=np.repeat(triangles,3,axis=1).ravel();cols=np.tile(triangles,(1,3)).ravel();n=len(coords)
    k=2*np.pi*frequency/c
    vals=(ke-k*k*me).ravel()
    # Consistent 1D FEM boundary mass and stiffness (not FV eigenvectors).
    K=np.zeros((nx+1,nx+1));M=K.copy()
    for ix,h in enumerate(np.diff(xe)):
        K[ix:ix+2,ix:ix+2]+=np.array([[1,-1],[-1,1]])/h
        M[ix:ix+2,ix:ix+2]+=h*np.array([[2,1],[1,2]])/6
    lam,phi=eigh(K,M);lam=np.maximum(lam,0);lam[0]=0
    width=xe[-1]-xe[0];phi*=np.sqrt(width)
    if phi[:,0].mean()<0:phi[:,0]*=-1
    beta=np.sqrt(k*k-lam+0j)
    radiation=(M@phi*(1j*beta))@(phi.T@M)/width
    inlet=ids[np.arange(nx+1)];outlet=ids[nz*(nx+1)+np.arange(nx+1)]
    if from_right:inlet,outlet=outlet,inlet
    rr=[rows];cc=[cols];vv=[vals.astype(complex)]
    for port in [inlet,outlet]:
        rr.append(np.repeat(port,nx+1));cc.append(np.tile(port,nx+1));vv.append(-radiation.ravel())
    A=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(n,n)).tocsc()
    rhs=np.zeros(n,complex);rhs[inlet]=-2j*k*(M@np.ones(nx+1))
    pressure=spsolve(A,rhs)
    t=phi.T@M@pressure[outlet]/width;r=phi.T@M@(pressure[inlet]-1)/width
    flux=beta.real/k;T=float(np.sum(flux*abs(t)**2));R=float(np.sum(flux*abs(r)**2))
    relative=t[0]*np.exp(-1j*k*(ze[-1]-ze[0]))
    field=np.full((nz+1,nx+1),np.nan+0j);field.ravel()[active]=pressure
    return dict(T=T,R=R,energy_error=abs(T+R-1),residual=float(np.linalg.norm(A@pressure-rhs)/np.linalg.norm(rhs)),
                amplitude=float(abs(t[0])),relative_phase_deg=float(np.angle(relative,deg=True)),pressure=field)
