import unittest
import numpy as np
from .benchmark_mesh import PreparedMesh,segmented_grid
from .solver import solve
from .paper_benchmark import geometry,CASES

class BenchmarkTests(unittest.TestCase):
    def test_uniform_assembly_matches_existing_solver(self):
        solid=np.zeros((30,25),bool);solid[10:15,4:9]=True
        for bc in ['rigid','periodic']:
            a=PreparedMesh(solid,np.arange(26)*.003,np.arange(31)*.003,bc).solve(1500)
            b=solve(solid,.003,.003,1500,transverse=bc)
            np.testing.assert_allclose(a.pressure,b.pressure,atol=2e-10)
            self.assertLess(a.balance_error,1e-10)

    def test_nonuniform_conservation_reciprocity_and_scaling(self):
        xe=segmented_grid([0,.026,.035,.049,.08],.003)
        ze=segmented_grid([0,.012,.025,.063,.09],.003)
        # Equal port-step construction.
        ze=np.r_[[-.006,-.003],ze,[.093,.096]]
        x=(xe[1:]+xe[:-1])/2;z=(ze[1:]+ze[:-1])/2
        solid=(x[None,:]>.026)&(x[None,:]<.049)&(z[:,None]>.025)&(z[:,None]<.063)
        for bc in ['rigid','periodic']:
            r=PreparedMesh(solid,xe,ze,bc).solve(1700)
            reverse=PreparedMesh(solid[::-1],xe,ze[-1]-ze[::-1],bc).solve(1700)
            scale=PreparedMesh(solid,xe*2,ze*2,bc).solve(850)
            self.assertLess(r.balance_error,1e-10)
            self.assertAlmostEqual(r.transmission,reverse.transmission,places=9)
            self.assertAlmostEqual(r.transmission,scale.transmission,places=9)

    def test_nonuniform_empty_and_closed(self):
        xe=segmented_grid([0,.031,.08],.002)
        ze=np.arange(40)*.002
        mask=np.zeros((len(ze)-1,len(xe)-1),bool)
        r=PreparedMesh(mask,xe,ze).solve()
        self.assertAlmostEqual(r.transmission,1,places=9)
        mask[17:20]=True;r=PreparedMesh(mask,xe,ze).solve()
        self.assertLess(r.transmission,1e-18)
        self.assertAlmostEqual(r.reflection,1,places=9)

    def test_submillimetre_slits_preserved(self):
        s,x,z,m=geometry(CASES[-1],.0005)
        iz=np.flatnonzero(s.sum(axis=1)==s.sum(axis=1).max())[0]
        widths=np.diff(x)
        runs=np.diff(np.r_[False,~s[iz],False].astype(int))
        actual=[widths[a:b].sum() for a,b in zip(np.flatnonzero(runs==1),np.flatnonzero(runs==-1))]
        np.testing.assert_allclose(actual,[.00058,.053,.00053],atol=1e-11)


class PortAndSymmetryTests(unittest.TestCase):
    def test_independent_ports_conserve_power_and_cancel(self):
        from .benchmark_ports import make_ports
        model=make_ports(1/3,.001)
        equal=model.solve(incident=[1,1,1])
        self.assertAlmostEqual(equal.transmission,1,places=9)
        r=model.solve(incident=np.exp(1j*np.array([-2,0,2])*np.pi/3))
        self.assertLess(r.transmission,1e-16)
        self.assertLess(r.balance_error,1e-10)
        r=model.solve(incident=[1,.4j,.7])
        self.assertLess(r.balance_error,1e-10)

    def test_supplement_s1_mirror_pair_matches_rigid_single(self):
        s,x,z,m=geometry(CASES[1],.001)
        one=PreparedMesh(s,x,z).solve()
        doubled=np.concatenate([s,s[:,::-1]],axis=1)
        xx=np.r_[x,2*x[-1]-x[-2::-1]]
        two=PreparedMesh(doubled,xx,z).solve()
        np.testing.assert_allclose(two.pressure[:,:s.shape[1]],one.pressure,atol=1e-8)
        self.assertAlmostEqual(two.transmission,one.transmission,places=9)


if __name__=='__main__':unittest.main()
