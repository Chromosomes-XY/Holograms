"""Phase-guided design diagnostics with explicit reference planes.

Paper equations 2--3 supply the relative phase prescription. The amplitude
target below uses a prescribed-pressure aperture surrogate, NOT a replication
of the authors' COMSOL port model and NOT a passive-device performance claim.
Only full passive geometry solves are used to report device R and T.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import minimize_scalar
from .geometry import rasterize
from .solver import solve


def phase_targets(openness):
    if not 0 < openness < 1:
        raise ValueError('Openness must be between zero and one')
    return np.array([-1.,0.,1.])*(1+openness)*np.pi/2


def ideal_target(openness=.75, widths=None):
    """Fix both side amplitudes at one; optimise central amplitude (no zero solution).

    At normal incidence with subwavelength pitch only the zeroth diffraction
    order propagates. Its pressure coefficient is sum(width_i * t_i).
    This solves the aperture cancellation target, not the passive inverse design.
    """
    weights=np.array([(1-openness)/2,openness,(1-openness)/2]
                     if widths is None else widths,dtype=float)
    if weights.shape != (3,) or np.any(weights<=0):
        raise ValueError('Three positive region widths required')
    phase=phase_targets(openness)
    def leakage(a):
        field=np.array([1.,a,1.])*np.exp(1j*phase)
        return float(abs(weights@field)**2)
    opt=minimize_scalar(leakage,bounds=(0.,1.),method='bounded',options={'xatol':1e-13})
    amps=np.array([1.,opt.x,1.])
    return amps*np.exp(1j*phase)


def channel_ranges(solid):
    """Contiguous air regions in the entrance straight guide, in x order."""
    active=np.flatnonzero(solid.any(axis=1))
    if not len(active):
        raise ValueError('No module found')
    first,last=int(active[0]),int(active[-1])
    air=~solid[first]
    transitions=np.diff(np.r_[False,air,False].astype(int))
    starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
    ranges=list(zip(starts,ends))
    if len(ranges)!=3:
        raise ValueError('Expected three distinct channels')
    return ranges,first,last


def isolated_channels(g,spacing=.001,frequency=1000.):
    """Extract each channel and solve in a rigid duct with an incident plane mode.

    Complex S21 is de-embedded from grid boundary centres to module faces.
    Per-channel inlet/outlet cross sections are equal. Phase reference is common
    physical module faces; side S21 is also given relative to the blank centre.
    """
    solid,dx,dz,_=rasterize(g,spacing)
    ranges,first,last=channel_ranges(solid)
    coefficients=[]; records=[]
    free_distance=(first-.5+len(solid)-last-1.5)*dz
    for a,b in ranges:
        r=solve(solid[:,a:b],dx,dz,frequency,transverse='rigid')
        if r.balance_error>1e-7 or r.residual>1e-7:
            raise RuntimeError('Isolated-channel solve failed verification')
        t=r.outgoing_modes[0]*np.exp(-1j*r.qz[0]*free_distance/dz)
        coefficients.append(t)
        records.append(dict(amplitude=float(abs(t)),phase_deg=float(np.angle(t,deg=True)),
                            transmission=r.transmission,reflection=r.reflection,
                            energy_error=r.balance_error,width_m=(b-a)*dx))
    for rec,t in zip(records,coefficients):
        rec['relative_phase_deg']=float(np.angle(t/coefficients[1],deg=True))
    return np.array(coefficients),records


def match_error(values,target):
    """Scale/overall-phase invariant complex-pattern error, 0=exact, 1=orthogonal.

    A vanishing field is assigned error one, avoiding undefined phase rewards.
    """
    norm=float(np.vdot(values,values).real)
    if norm<1e-16:
        return 1.
    scale=np.vdot(target,values)/np.vdot(target,target)
    return float(np.clip(np.vdot(values-scale*target,values-scale*target).real/norm,0,1))


def directional_channels(g,solid,r,frequency=1000.):
    """Separate forward/backward fundamental pressure in each exit guide.

    A least-squares fit of mean pressure over a uniform guide section removes
    the backward wave. Coefficients refer to the physical outlet face. These
    are in-situ modal amplitudes under the common external illumination, NOT
    isolated S21. Net pressure at the mouth is retained as a separate diagnostic.
    """
    ranges,_,last=channel_ranges(solid)
    first=last
    while first>0 and np.array_equal(solid[first-1],solid[last]):
        first-=1
    first=max(first,last-int(round(.025/r.dz))+1)
    indices=np.arange(first,last+1)
    if len(indices)<4:
        raise ValueError('Need at least four uniform guide rows for modal fitting')
    offsets=(indices-last-.5)*r.dz
    q=2*np.arcsin(np.pi*frequency*r.dz/343.)
    fit=np.column_stack([np.exp(1j*q*offsets/r.dz),np.exp(-1j*q*offsets/r.dz)])
    forward=[];backward=[];residual=[]
    for a,b in ranges:
        mean=r.pressure[indices,a:b].mean(axis=1)
        coef=np.linalg.lstsq(fit,mean,rcond=None)[0]
        forward.append(coef[0]);backward.append(coef[1])
        residual.append(float(np.linalg.norm(fit@coef-mean)/max(np.linalg.norm(mean),1e-20)))
    forward=np.array(forward);backward=np.array(backward)
    phase=np.angle(forward*np.conj(forward[1]))
    target=ideal_target(g.central_fraction,[(b-a)*r.dx/g.pitch for a,b in ranges])
    return dict(forward_amplitudes=abs(forward).tolist(),backward_amplitudes=abs(backward).tolist(),
                forward_phase_deg=np.rad2deg(phase).tolist(),
                forward_phase_errors_deg=np.rad2deg(np.angle(np.exp(1j*(phase-phase_targets(g.central_fraction))))).tolist(),
                forward_target_match_error=match_error(forward,target),fit_relative_residual=residual,
                reference='forward and backward fundamental-mode pressure at outlet face, under assembled illumination')


def diagnostics(g,solid,r,frequency=1000.):
    """Measure actual exit pressure pattern and propagating/evanescent spectrum.

    Regional pressure is averaged on the first homogeneous row after the outlet
    and de-embedded half a grid spacing to the mouth. It is the assembled field,
    NOT an independently incident single-channel transmission coefficient.
    """
    ranges,first,last=channel_ranges(solid)
    basis=r.transverse_basis
    # Last boundary centre -> physical downstream face. High-k evanescent
    # components cannot be safely back-propagated over the exterior padding.
    # Instead extract modes on the FIRST air row, where the field is outgoing.
    at_row=r.pressure[last+1]
    coeff=basis.conj().T@at_row/len(r.kx)
    mouth=coeff*np.exp(-.5j*r.qz)
    mouth_field=basis@mouth
    regional=np.array([mouth_field[a:b].mean() for a,b in ranges])
    widths=np.array([(b-a)*r.dx/g.pitch for a,b in ranges])
    target=ideal_target(g.central_fraction,widths)
    relative=np.angle(regional*np.conj(regional[1]))
    desired=phase_targets(g.central_fraction)
    errors=np.angle(np.exp(1j*(relative-desired)))
    propagating=abs(r.qz.imag)<1e-10
    flux=np.where(propagating,np.sin(r.qz.real)/r.dz,0.)
    per_mode=abs(mouth)**2*flux/flux[0]
    evan_mask=~propagating
    distances=np.array([0.,.025,.05,.1,.2])
    evan_rms=np.sqrt(np.sum(abs(mouth[evan_mask])[None,:]**2*
                           np.exp(-2*np.outer(distances,r.qz[evan_mask].imag/r.dz)),axis=1))
    prop_rms=float(np.linalg.norm(mouth[propagating]))
    near_evan_fraction=float(np.sum(abs(mouth[evan_mask])**2)/max(np.sum(abs(mouth)**2),1e-30))
    mode_order=np.rint(np.fft.fftfreq(len(mouth))*len(mouth)).astype(int)
    order1=(abs(mode_order)==1)&evan_mask
    positive=np.sum(abs(mouth[(mode_order==1)&evan_mask])**2)
    negative=np.sum(abs(mouth[(mode_order==-1)&evan_mask])**2)
    phase_valid=bool(np.min(abs(regional))>1e-5)
    return dict(
        transmission=r.transmission,reflection=r.reflection,
        tl_db=float(-10*np.log10(max(r.transmission,1e-16))),
        energy_error=r.balance_error,residual=r.residual,
        regional_amplitudes=abs(regional).tolist(),
        regional_phase_deg=np.rad2deg(relative).tolist(),
        target_amplitudes=abs(target).tolist(),target_phase_deg=np.rad2deg(desired).tolist(),
        regional_phase_errors_deg=np.rad2deg(errors).tolist(),phase_valid=phase_valid,
        target_match_error=match_error(regional,target),
        aperture_width_fractions=widths.tolist(),
        near_evanescent_pressure_squared_fraction=near_evan_fraction,
        first_order_evanescent_rms=float(np.linalg.norm(mouth[order1])),
        first_order_directional_imbalance=float((positive-negative)/max(positive+negative,1e-30)),
        propagating_pressure_rms=prop_rms,
        evanescent_rms_distance_m=distances.tolist(),evanescent_pressure_rms=evan_rms.tolist(),
        modes=[dict(order=int(n),kx=float(kx),pressure_amplitude=float(abs(a)),
                    propagating=bool(pr),transmitted_power=float(power),
                    decay_length_m=float(r.dz/q.imag) if q.imag>1e-10 else None)
               for n,kx,a,pr,power,q in zip(mode_order,r.kx,mouth,propagating,per_mode,r.qz)],
        notes='Evanescent fraction is pressure-squared, NOT transmitted power. '
              'Exit regional averages are NOT isolated-channel S21. '
              'Evanescent fields alone do not prove a bound surface eigenmode.')


def evaluate_phase(g,spacing=.001,frequency=1000.,angle=0.):
    solid,dx,dz,metrics=rasterize(g,spacing)
    r=solve(solid,dx,dz,frequency,angle)
    d=diagnostics(g,solid,r,frequency)
    if d['energy_error']>1e-7 or d['residual']>1e-7:
        raise RuntimeError('Full solve failed conservation/residual checks')
    return d,metrics
