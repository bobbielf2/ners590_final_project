import unittest
import numpy as np
from quadr import quadr

class TestQuadr(unittest.TestCase):

    def test_badInput(self):
        with self.assertRaises(Exception):
            s = {}
            s = quadr(s) # not enough input data to generate geometry
        with self.assertRaises(Exception):
            s = {}
            s['x'] = np.array([3])
            s = quadr(s, 100) # input data has conflict
        
        # override inconsistent data when possible
        s = {}
        s['Z'] = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)     # starfish param
        s['x'] = np.array([3])
        n = 100
        s = quadr(s, n) # should override s['x']
        self.assertTrue(s['x'].size == n)

    def test_normalVec(self):
        Z = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)   # starfish param
        s = {}
        n = 100
        s['x'] = Z(np.arange(0,n)/n*2*np.pi)
        s = quadr(s,n)
        # Now check that normals from spectral differentiation are accurate:
        Zp = lambda s: -1.5 * np.sin(5*s) * np.exp(1j*s) + 1j * Z(s)        # Z' formula
        s['Z'] = Z
        s['Zp'] = Zp
        t = quadr(s)
        # error in normal vec, should be small
        self.assertTrue(np.linalg.norm(t['nx']-s['nx']) <= np.finfo(float).eps)


if __name__ == '__main__':
    unittest.main()


