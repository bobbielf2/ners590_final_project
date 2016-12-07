import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import path
from quadr import quadr
from LapSLPmatrix import LapSLPmatrix

def test_SLPn():
    # test data dir
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/TestData/"
    circle   = sio.loadmat(dir_path+'circle.mat')
    x        = circle['x']
    N, M     = x.shape
    
    # set up source
    s = {}
    for l in range(0,M):
        s_temp      = {}
        s_temp['x'] = x[:,l][:,np.newaxis]
        s_temp      = quadr(s_temp,N)
        s[str(l)]  = s_temp


    # set up target
    nx       = 100
    gx       = np.arange(1,nx+1)/nx
    ny       = 100
    gy       = np.arange(1,ny+1)/ny     # set up plotting
    xx, yy   = np.meshgrid(gx,gy)
    zz       = xx + 1j*yy
    
    t        = {}
    ii       = np.ones((nx*ny, ), dtype=bool)
    for l in range(0,M):
        s_temp      = s[str(l)]
        p          = path.Path(np.vstack((np.real(s_temp['x']).T,np.imag(s_temp['x']).T)).T)
        ii         = (~p.contains_points(np.vstack((np.real(zz).flatten('F'),np.imag(zz).flatten('F'))).T))&ii

    t['x']   = zz.flatten('F')[ii][np.newaxis].T
    
    
    # multipole evaluation
    u        = 0*(1+1j)*zz
    idx      = ii.reshape(ny,nx,order='F')
    for l in range(0,M):
        s_temp     = s[str(l)]
        A          = LapSLPmatrix(t,s_temp,0)
        tau        = np.sin(2*np.pi*np.real(s_temp['x'])) + np.cos(np.pi*np.imag(s_temp['x']))
        u_temp     = A.dot(tau)
        u.T[idx.T] = u.T[idx.T] + u_temp.flatten()
        if np.mod(l,25) == 0:
            fig = plt.figure()
            logerr = plt.imshow(np.real(u),aspect=nx/ny, interpolation='none')
            fig.colorbar(logerr)
            plt.grid(True)
            plt.show()
    
    fig = plt.figure()
    logerr = plt.imshow(np.real(u),aspect=nx/ny, interpolation='none')
    fig.colorbar(logerr)
    plt.grid(True)
    plt.show()
        
    
    
    
if __name__ == '__main__':
    test_SLPn()
    