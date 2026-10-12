import unittest
from dataclasses import replace
import numpy as np
from .solver import solve
from .geometry import Geometry, rasterize
from .phase_design import phase_targets,ideal_target,isolated_channels,diagnostics,match_error,directional_channels


class PhaseDesignTests(unittest.TestCase):
    def test_paper_phase_and_nonzero_amplitude_target(self):
        np.testing.assert_allclose(np.rad2deg(phase_targets(.75)),[-157.5,0,157.5])
        t=ideal_target(.75)
        self.assertAlmostEqual(abs(t[0]),1)
        self.assertAlmostEqual(abs(t[2]),1)
        self.assertAlmostEqual(abs(t[1]),.307959844,places=8)
        self.assertLess(abs(np.array([.125,.75,.125])@t),1e-8)
        self.assertAlmostEqual(match_error(2j*t,t),0,places=12)
        self.assertEqual(match_error(np.zeros(3),t),1)

    def test_rigid_duct_plane_wave_and_reciprocity(self):
        empty=np.zeros((70,20),bool)
        r=solve(empty,.002,.002,transverse='rigid')
        self.assertAlmostEqual(r.transmission,1,places=10)
        np.testing.assert_allclose(r.downstream([0])[0],r.pressure[-1],atol=1e-10)
        empty[25:30,:5]=True;empty[35:40,-8:]=True
        r=solve(empty,.002,.002,transverse='rigid')
        rev=solve(empty[::-1],.002,.002,transverse='rigid')
        self.assertLess(r.balance_error,1e-10)
        self.assertAlmostEqual(r.transmission,rev.transmission,places=10)

    def test_deembedding_and_modal_flux(self):
        g=Geometry()
        t,records=isolated_channels(g,.002)
        q=2*np.arcsin(.002*np.pi*1000/343)
        np.testing.assert_allclose(t[1],np.exp(1j*q*(86+2*34)),atol=1e-10)
        solid,dx,dz,_=rasterize(g,.002)
        r=solve(solid,dx,dz)
        d=diagnostics(g,solid,r)
        self.assertAlmostEqual(sum(m['transmitted_power'] for m in d['modes']),r.transmission,places=9)
        self.assertTrue(np.all(np.diff(d['evanescent_pressure_rms'])<=0))
        self.assertEqual(d['target_phase_deg'],[-157.5,0,157.5])

    def test_known_evanescent_component(self):
        solid,dx,dz,_=rasterize(Geometry(),.002)
        r=solve(solid,dx,dz)
        last=np.flatnonzero(solid.any(axis=1))[-1]
        r.pressure[last+1]=r.transverse_basis[:,1]*np.exp(.5j*r.qz[1])
        d=diagnostics(Geometry(),solid,r)
        expected=np.exp(-np.array(d['evanescent_rms_distance_m'])*r.qz[1].imag/dz)
        np.testing.assert_allclose(d['evanescent_pressure_rms'],expected,atol=1e-10)
        self.assertLess(d['propagating_pressure_rms'],1e-10)

    def test_rib_translation_preserves_isolated_transmission(self):
        g=Geometry()
        base,_=isolated_channels(g,.002)
        shifted=replace(g,offset_left=.010,offset_right=-.008)
        moved,_=isolated_channels(shifted,.002)
        np.testing.assert_allclose(moved,base,atol=1e-9)
        _,_,_,m=rasterize(shifted,.002)
        self.assertGreaterEqual(m['central_clear_fraction'],.75)

    def test_directional_fit_separates_reflected_wave(self):
        g=Geometry();solid,dx,dz,_=rasterize(g,.002)
        r=solve(solid,dx,dz)
        q=r.qz[0].real
        last=np.flatnonzero(solid.any(axis=1))[-1]
        offsets=(np.arange(len(solid))-last-.5)
        artificial=2*np.exp(1j*q*offsets)+.3j*np.exp(-1j*q*offsets)
        r.pressure=np.broadcast_to(artificial[:,None],solid.shape).copy()
        r.pressure[solid]=np.nan
        d=directional_channels(g,solid,r)
        np.testing.assert_allclose(d['forward_amplitudes'],2,atol=1e-10)
        np.testing.assert_allclose(d['backward_amplitudes'],.3,atol=1e-10)
        self.assertLess(max(d['fit_relative_residual']),1e-10)


if __name__=='__main__':
    unittest.main()
