import unittest
import numpy as np
from .independent_fem import solve_fem

class IndependentFEMTests(unittest.TestCase):
    def test_uniform_guide_phase_and_power(self):
        xe=np.linspace(0,.025,51);ze=np.linspace(0,.09,181)
        r=solve_fem(np.zeros((180,50),bool),xe,ze)
        self.assertLess(abs(r['T']-1),1e-7)
        self.assertLess(abs(r['relative_phase_deg']),.01)
        self.assertLess(r['energy_error'],1e-9)

    def test_closed_guide(self):
        xe=np.linspace(0,.025,26);ze=np.linspace(0,.09,91);s=np.zeros((90,25),bool);s[40:45]=True
        r=solve_fem(s,xe,ze)
        self.assertLess(r['T'],1e-18);self.assertAlmostEqual(r['R'],1,places=9)

    def test_reciprocity_and_scaling_with_obstacle(self):
        xe=np.linspace(0,.025,51);ze=np.linspace(0,.09,181);s=np.zeros((180,50),bool);s[65:88,:22]=True
        a=solve_fem(s,xe,ze);b=solve_fem(s,xe,ze,from_right=True);c=solve_fem(s,xe*2,ze*2,frequency=1000)
        self.assertLess(a['energy_error'],1e-9)
        self.assertAlmostEqual(a['T'],b['T'],places=8)
        self.assertAlmostEqual(a['T'],c['T'],places=9)

if __name__=='__main__':unittest.main()
