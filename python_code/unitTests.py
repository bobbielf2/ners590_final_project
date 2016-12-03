import unittest
import numpy as np
import scipy.io as sio
from quadr import quadr
from matplotlib import path
from LapSLPmatrix import LapSLPmatrix

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
        
        
class TestLapSLP(unittest.TestCase):
    
    def setUp(self):
        
        N              = 500
        # source: starfish domain
        a              = 0.3
        w              = 5
        R              = lambda t: (1 + a*np.cos(w*t))*1
        Rp             = lambda t: -w*a*np.sin(w*t)
        Rpp            = lambda t: -w*w*a*np.cos(w*t)
        self.s         = {}
        self.s['Z']    = lambda t: R(t)*np.exp(1j*t)
        self.s['Zp']   = lambda t: (Rp(t) + 1j*R(t))*np.exp(1j*t)
        self.s['Zpp']  = lambda t: (Rpp(t) + 2j*Rp(t) - R(t))*np.exp(1j*t)
        self.s         = quadr(self.s,N)
        
        # target
        nx             = 100
        gx             = (np.arange(1,nx+1)/nx*2-1)*1.5
        ny             = 100
        gy             = (np.arange(1,ny+1)/ny*2-1)*1.5     # set up plotting
        xx, yy         = np.meshgrid(gx,gy)
        zz             = xx + 1j*yy
        
        self.t         = {}
        p              = path.Path(np.vstack((np.real(self.s['x']).T,np.imag(self.s['x']).T)).T)
        ii             = ~p.contains_points(np.vstack((np.real(zz).flatten('F'),np.imag(zz).flatten('F'))).T)
        self.t['x']    = zz.flatten('F')[ii][np.newaxis].T
        
        # bad target
        self.tbad      = {}
        self.tbad['x'] = self.s['x']
        
        
        # precision 
        self.eps       = 10**(-14)      # An won't accept smaller precision...
        
        # test data dir
        import os 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dir       = dir_path + "/TestData/"
    
        
    # check that when input target is bad (normal derivative wasn't provided)
    def test_badInput(self):
        with self.assertRaises(Exception):
            _, An = LapSLPmatrix(self.tbad,self.s,0,1)
        
    
    # check that SLPmatrix A in python and matlab are the same        
    def test_matlabEqualA(self):
        test      = sio.loadmat(self.dir+'testA.mat')
        A         = LapSLPmatrix(self.t,self.s,0,0)              
        self.assertTrue(np.abs(np.max(A-test['A'])) <= self.eps)
    
    # check that normal derivative An in python and matlab are the same
    def test_matlabEqualAn(self):
        test      = sio.loadmat(self.dir+'testAn.mat')
        _, An     = LapSLPmatrix(self.s,self.s,0,1)
        self.assertTrue(np.abs(np.max(An-test['An'])) <= self.eps)


if __name__ == '__main__':
    unittest.main()


