"""Interface-aligned rectangular finite volumes for paper benchmarking.

Exp(-i wt), rigid solids, exact discrete outgoing modal ports. Nonuniform cells
preserve sub-mm slit dimensions rather than snapping them to a uniform grid.
This is an independent assembly path, checked against solver.py on uniform grids.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def segmented_grid(points, maximum_step):
    points=np.unique(np.round(points,12))
    return np.concatenate([np.linspace(a,b,max(1,int(np.ceil((b-a)/maximum_step-1e-9)))+1)[:-1]
                           for a,b in zip(points[:-1],points[1:])]+[points[-1:]])


def graded_grid(points,maximum_step,edge_step):
    """Symmetric refinement near rectangle edges; fixed geometry at every level."""
    points=np.unique(np.round(points,12));out=[]
    for a,b in zip(points[:-1],points[1:]):
        left=[];remaining=b-a;step=min(edge_step,maximum_step)
        while remaining>2*step:
            left.append(step);remaining-=2*step;step=min(step*1.35,maximum_step)
        if left and remaining<edge_step/2:remaining+=2*left.pop()
        n=max(1,int(np.ceil(remaining/maximum_step)))
        widths=left+[remaining/n]*n+left[::-1]
        out.extend((a+np.r_[0,np.cumsum(widths)[:-1]]).tolist())
    return np.r_[out,points[-1]]


def transverse_modes(xedges, boundary='rigid'):
    w=np.diff(xedges);x=(xedges[1:]+xedges[:-1])/2;n=len(w)
    L=np.zeros((n,n))
    for i in range(n-1):
        a=1/(x[i+1]-x[i]);L[i,i]+=a;L[i+1,i+1]+=a;L[i,i+1]-=a;L[i+1,i]-=a
    if boundary=='periodic':
        a=2/(w[0]+w[-1]);L[0,0]+=a;L[-1,-1]+=a;L[0,-1]-=a;L[-1,0]-=a
    elif boundary!='rigid':raise ValueError(boundary)
    lam,u=eigh(L/np.sqrt(w[:,None]*w[None,:]))
    lam=np.maximum(lam,0);lam[0]=0
    phi=u/np.sqrt(w[:,None])*np.sqrt(w.sum())
    if phi[:,0].mean()<0:phi[:,0]*=-1
    return lam,phi


@dataclass
class MeshResult:
    pressure: np.ndarray
    transmission: float
    reflection: float
    residual: float
    balance_error: float
    forward: np.ndarray
    reflected: np.ndarray
    phi: np.ndarray
    q: np.ndarray
    port_step: float
    xedges: np.ndarray
    zedges: np.ndarray

    def extend(self,distance,upstream=False):
        d=np.asarray(distance)
        coeff=self.reflected if upstream else self.forward
        wave=(np.exp(1j*np.outer(d,self.q/self.port_step))*coeff)@self.phi.T
        if upstream:wave+=np.exp(-1j*d[:,None]*self.q[0]/self.port_step)
        return wave


class PreparedMesh:
    def __init__(self,solid,xedges,zedges,boundary='rigid',blocked_x=None,inlet_groups=None):
        self.solid=solid=np.asarray(solid,bool);self.xedges=np.asarray(xedges);self.zedges=np.asarray(zedges)
        self.wx=np.diff(xedges);self.wz=np.diff(zedges)
        self.x=(xedges[1:]+xedges[:-1])/2;self.z=(zedges[1:]+zedges[:-1])/2
        nz,nx=solid.shape
        assert (nz,nx)==(len(self.wz),len(self.wx))
        assert not solid[[0,-1]].any()
        if not np.isclose(self.wz[0],self.wz[-1],rtol=1e-9,atol=1e-12):raise ValueError('Unequal port steps')
        self.h=self.wz[0];self.lam,self.phi=transverse_modes(self.xedges,boundary)
        ids=np.full(solid.shape,-1,int);ids[~solid]=np.arange((~solid).sum());self.ids=ids
        rows=[];cols=[];vals=[];diagonal=np.zeros((~solid).sum())
        def connect(a,b,g):
            a=a.ravel();b=b.ravel();g=np.broadcast_to(g,a.shape if np.ndim(g)==0 else np.shape(g)).ravel()
            ok=(a>=0)&(b>=0);a=a[ok];b=b[ok];g=g[ok]
            rows.extend([a,b]);cols.extend([b,a]);vals.extend([g,g])
            np.add.at(diagonal,a,-g);np.add.at(diagonal,b,-g)
        gx=self.wz[:,None]/np.diff(self.x)[None,:]
        if blocked_x is not None:gx=np.where(blocked_x,0,gx)
        connect(ids[:,:-1],ids[:,1:],gx)
        connect(ids[:-1],ids[1:],self.wx[None,:]/np.diff(self.z)[:,None])
        if boundary=='periodic':connect(ids[:,0],ids[:,-1],self.wz/(.5*(self.wx[0]+self.wx[-1])))
        rows.append(np.arange(len(diagonal)));cols.append(rows[-1]);vals.append(diagonal)
        self.base=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(diagonal),len(diagonal))).tocsc()
        self.mass=(self.wz[:,None]*self.wx[None,:])[~solid]
        self.port_ids=[ids[0],ids[-1]]
        self.inlet_groups=inlet_groups
        self.inlet_modes=[]
        if inlet_groups is not None:
            for a,b in inlet_groups:
                lam,phi=transverse_modes(self.xedges[a:b+1],'rigid')
                self.inlet_modes.append((a,b,lam,phi))

    def solve(self,frequency=2000,c=343,incident=None):
        if not np.isfinite([frequency,c]).all() or min(frequency,c)<=0:
            raise ValueError('Positive finite frequency and sound speed required')
        if incident is not None and (not np.isfinite(incident).all() or not np.any(np.abs(incident)>0)):
            raise ValueError('Nonzero finite incident amplitudes required')
        k=2*np.pi*frequency/c
        q=2*np.arcsin(.5*self.h*np.sqrt(k*k-self.lam+0j));q=np.where(q.imag<0,-q,q)
        roots=np.exp(1j*q)
        radiation=(self.phi*roots)@(self.phi.T*self.wx[None,:])/self.wx.sum()
        port=(self.wx[:,None]/self.h)*(radiation-np.eye(len(self.wx)))
        rr=[];cc=[];vv=[]
        for ids in (self.port_ids if self.inlet_groups is None else [self.port_ids[1]]):
            rr.append(np.repeat(ids,len(ids)));cc.append(np.tile(ids,len(ids)));vv.append(port.ravel())
        inlet_data=[]
        if self.inlet_groups is not None:
            if incident is None:incident=np.ones(len(self.inlet_groups),complex)
            if len(incident)!=len(self.inlet_groups):raise ValueError('One incident amplitude per inlet')
            for (a,b,lam,phi),amp in zip(self.inlet_modes,incident):
                qq=2*np.arcsin(.5*self.h*np.sqrt(k*k-lam+0j));qq=np.where(qq.imag<0,-qq,qq)
                root=np.exp(1j*qq);w=self.wx[a:b]
                rad=(phi*root)@(phi.T*w[None,:])/w.sum()
                ids=self.port_ids[0][a:b]
                rr.append(np.repeat(ids,len(ids)));cc.append(np.tile(ids,len(ids)));vv.append(((w[:,None]/self.h)*(rad-np.eye(len(w)))).ravel())
                inlet_data.append((a,b,phi,qq,root,amp))
        ii=np.arange(len(self.mass));rr.append(ii);cc.append(ii);vv.append(k*k*self.mass)
        a=self.base+coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=self.base.shape).tocsc()
        b=np.zeros(len(self.mass),complex);b[self.port_ids[0]]=-(1/roots[0]-roots[0])*self.wx/self.h
        if inlet_data:
            b[:]=0
            for lo,hi,ph,qq,rt,amp in inlet_data:
                b[self.port_ids[0][lo:hi]]=-(1/rt[0]-rt[0])*self.wx[lo:hi]/self.h*amp
        p=spsolve(a,b);field=np.full(self.solid.shape,np.nan+0j);field[~self.solid]=p
        forward=self.phi.T@(self.wx*field[-1])/self.wx.sum()
        reflected=self.phi.T@(self.wx*(field[0]-1))/self.wx.sum()
        flux=np.where(abs(q.imag)<1e-9,np.sin(q.real)/self.h,0);weights=flux/flux[0]
        T=float(np.sum(weights*abs(forward)**2));R=float(np.sum(weights*abs(reflected)**2))
        if inlet_data:
            pin=0.;pref=0.
            for lo,hi,ph,qq,rt,amp in inlet_data:
                w=self.wx[lo:hi];ref=ph.T@(w*(field[0,lo:hi]-amp))/w.sum()
                fl=np.where(abs(qq.imag)<1e-9,np.sin(qq.real)/self.h,0)
                pin+=w.sum()*fl[0]*abs(amp)**2
                pref+=w.sum()*np.sum(fl*abs(ref)**2)
            T=float(np.sum(flux*abs(forward)**2)*self.wx.sum()/pin);R=float(pref/pin)
        residual=float(np.linalg.norm(a@p-b)/np.linalg.norm(b))
        return MeshResult(field,T,R,residual,abs(T+R-1),forward,reflected,self.phi,q,self.h,self.xedges,self.zedges)
