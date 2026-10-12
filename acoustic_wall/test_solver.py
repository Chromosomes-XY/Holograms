import unittest
from dataclasses import replace
import numpy as np
from .solver import solve
from .geometry import Geometry, rasterize


class ScatteringTests(unittest.TestCase):
    def test_empty_domain_and_angular_spectrum(self):
        for angle in (0,30,-40):
            r=solve(np.zeros((40,32),bool),.004,.004,angle_deg=angle)
            self.assertAlmostEqual(r.transmission,1,places=10)
            self.assertLess(r.reflection,1e-20)
            np.testing.assert_allclose(r.downstream([0])[0],r.pressure[-1],atol=1e-10)
            np.testing.assert_allclose(abs(r.downstream([.1,.3])),1,atol=1e-10)

    def test_closed_wall(self):
        solid=np.zeros((40,32),bool); solid[20:22]=True
        r=solve(solid,.004,.004)
        self.assertAlmostEqual(r.reflection,1,places=10)
        self.assertLess(r.transmission,1e-20)

    def test_conservation_reciprocity_and_scaling(self):
        solid=np.zeros((48,40),bool)
        solid[18:22,5:15]=True; solid[25:29,20:30]=True
        r=solve(solid,.004,.004)
        reverse=solve(solid[::-1],.004,.004)
        scale=solve(solid,.008,.008,frequency=500)
        self.assertLess(r.balance_error,1e-10)
        self.assertLess(r.residual,1e-10)
        self.assertAlmostEqual(r.transmission,reverse.transmission,places=9)
        self.assertAlmostEqual(r.transmission,scale.transmission,places=10)

    def test_multiple_propagating_orders(self):
        solid=np.zeros((45,100),bool); solid[20:24,5:40]=True
        r=solve(solid,.004,.004,frequency=1600,angle_deg=20)
        self.assertGreater(np.count_nonzero(abs(r.qz.imag)<1e-10),1)
        self.assertLess(r.balance_error,1e-10)

    def test_open_fraction_and_resolution(self):
        _,_,_,m=rasterize(Geometry())
        self.assertGreaterEqual(m['central_clear_fraction'],.75)
        self.assertGreaterEqual(m['total_line_of_sight_fraction'],.75)
        with self.assertRaises(ValueError):
            rasterize(replace(Geometry(),slit_left=.001))

    def test_refinement_for_rectangular_obstacle(self):
        transmissions=[]
        for h in (.004,.002,.001):
            x=(np.arange(round(.16/h))+.5)*h
            z=(np.arange(round(.16/h))+.5)*h
            solid=(abs(z[:,None]-.08)<.016)&(abs(x[None,:]-.08)<.04)
            transmissions.append(solve(solid,h,h).transmission)
        self.assertLess(abs(transmissions[2]-transmissions[1]),abs(transmissions[1]-transmissions[0]))
        self.assertLess(abs(transmissions[2]-transmissions[1]),.01)


if __name__=='__main__':
    unittest.main()
