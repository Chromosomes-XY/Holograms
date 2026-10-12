"""Oblique-incidence check on the same interface-aligned FV mesh.

Bloch seam phase and weighted Hermitian transverse eigensystem; select the
incident mode by overlap with exp(i k sin(theta) x), not eigenvalue order.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from .benchmark_mesh import PreparedMesh

def solve_bloch(s,x,z,frequency,angle,c=343):
 mesh=PreparedMesh(s,x,z,boundary='periodic');w=mesh.wx;xc=mesh.x;P=w.sum();n=len(w);k=2*np.pi*frequency/c;alpha=k*np.sin(np.deg2rad(angle));phase=np.exp(1j*alpha*P)
 L=np.zeros((n,n),complex)
 for i,g in enumerate(1/np.diff(xc)):
  L[i,i]+=g;L[i+1,i+1]+=g;L[i,i+1]-=g;L[i+1,i]-=g
 g=2/(w[0]+w[-1]);L[0,0]+=g;L[-1,-1]+=g;L[0,-1]-=g/phase;L[-1,0]-=g*phase
 lam,u=eigh(L/np.sqrt(w[:,None]*w[None,:]));lam=np.maximum(lam,0);phi=u/np.sqrt(w[:,None])*np.sqrt(P)
 plane=np.exp(1j*alpha*xc);idx=int(np.argmax(abs(phi.conj().T@(w*plane)/P)));inc=phi[:,idx]
 q=2*np.arcsin(.5*mesh.h*np.sqrt(k*k-lam+0j));q=np.where(q.imag<0,-q,q);roots=np.exp(1j*q)
 assert abs(q[idx].imag)<1e-9
 rad=(phi*roots)@(phi.conj().T*w[None,:])/P;port=(w[:,None]/mesh.h)*(rad-np.eye(n))
 rr=[];cc=[];vv=[]
 for ids in mesh.port_ids:
  rr.append(np.repeat(ids,n));cc.append(np.tile(ids,n));vv.append(port.ravel())
 ids=mesh.ids;left=ids[:,0];right=ids[:,-1];ok=(left>=0)&(right>=0);conductance=mesh.wz/(.5*(w[0]+w[-1]))
 rr.extend([left[ok],right[ok]]);cc.extend([right[ok],left[ok]]);vv.extend([conductance[ok]*(1/phase-1),conductance[ok]*(phase-1)])
 ii=np.arange(len(mesh.mass));rr.append(ii);cc.append(ii);vv.append(k*k*mesh.mass)
 A=mesh.base+coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=mesh.base.shape).tocsc()
 rhs=np.zeros(len(mesh.mass),complex);rhs[mesh.port_ids[0]]=-(1/roots[idx]-roots[idx])*w/mesh.h*inc
 p=spsolve(A,rhs);forward=phi.conj().T@(w*p[mesh.port_ids[1]])/P;reflected=phi.conj().T@(w*(p[mesh.port_ids[0]]-inc))/P
 flux=np.where(abs(q.imag)<1e-9,np.sin(q.real)/mesh.h,0);T=float(np.sum(flux*abs(forward)**2)/flux[idx]);R=float(np.sum(flux*abs(reflected)**2)/flux[idx])
 return dict(frequency_hz=frequency,angle_deg=angle,T=T,R=R,energy_error=abs(T+R-1),residual=float(np.linalg.norm(A@p-rhs)/np.linalg.norm(rhs)),incident_mode_index=idx)
