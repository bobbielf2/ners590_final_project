#!/usr/bin/env python3
"""
Created on Mon Nov 28 11:28:10 2016

@author: haizhu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from mpl_toolkits.mplot3d import Axes3D
from quadr import quadr

def LapSLPmatrix(t,s,a = 0,der = 0):
    
    N = s['x'].size
    M = t['x'].size
    d = np.tile(t['x'],(1,N)) - np.tile(s['x'].T + a,(M,1))
    A = -1/2/np.pi*np.log(np.abs(d))*np.tile(s['w'].T,(M,1))
    
    # return A
    if der == 0:
        return A
    elif 'nx' in t:
        # return An
        nx = np.tile(-t['nx'],(1,N))
        An = 1/2/np.pi*np.real(nx/d)
        if ( N == M ) & (np.max(np.abs(s['x']+a-t['x']))<np.exp(-14)):
            Andim = An.shape[0]
            di = np.diag_indices(Andim)
            An[di] = -s['cur'].T/4/np.pi
        An = An*np.tile(s['w'].T,(M,1))
        return A, An
    else:
        raise Exception("t['nx'] missing to compute normal derivative of A")

    
    
def test_laplaceeval():

    import matplotlib.pyplot as plt
    from matplotlib import path
    from mpl_toolkits.mplot3d import Axes3D
    from quadr import quadr

    side     = 'e'
    lptype   = 's'
    N        = 1500
    
    # set up source and target
    
    # source: starfish domain
    a        = 0.3
    w        = 5
    R        = lambda t: (1 + a*np.cos(w*t))*1
    Rp       = lambda t: -w*a*np.sin(w*t)
    Rpp      = lambda t: -w*w*a*np.cos(w*t)
    s        = {}
    s['Z']   = lambda t: R(t)*np.exp(1j*t)
    s['Zp']  = lambda t: (Rp(t) + 1j*R(t))*np.exp(1j*t)
    s['Zpp'] = lambda t: (Rpp(t) + 2j*Rp(t) - R(t))*np.exp(1j*t)
    s = quadr(s,N)
    
    # target
    nx       = 100
    gx       = (np.arange(1,nx+1)/nx*2-1)*1.5
    ny       = 100
    gy       = (np.arange(1,ny+1)/ny*2-1)*1.5     # set up plotting
    xx, yy   = np.meshgrid(gx,gy)
    zz       = xx + 1j*yy
    
    t        = {}
    p        = path.Path(np.vstack((np.real(s['x']).T,np.imag(s['x']).T)).T)
    ii       = ~p.contains_points(np.vstack((np.real(zz).flatten('F'),np.imag(zz).flatten('F'))).T)
    t['x']   = zz.flatten('F')[ii][np.newaxis].T
    
    # generate exact solution
    uexa     = np.nan*(1+1j)*zz
    A        = LapSLPmatrix(t,s,0)              # computation is arranged columnwised
    tau      = np.sin(np.abs(s['x'])) + np.cos(np.abs(s['x']))
    u_temp   = A.dot(tau)                       # u_temp is columnwised
    idx      = ii.reshape(ny,nx,order='F')      # uexa[idx] is rowwised
    uexa.T[idx.T] = u_temp.flatten()            # !!!!!!!!!!!!!!!!!!!!!
    uexa[~idx] = 0
    
    # plot exact solution 
    fig1 = plt.figure(1)
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(xx, yy, np.real(uexa), cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
    fig1.colorbar(surf)
    plt.show()
    
    # verify SLP matrix
    Nn = 5
    err = np.nan*np.ones(Nn)
    for NN in range(1,Nn+1):
        N = 100*NN
        s = quadr(s,N)
        
        u = np.nan*(1+1j)*zz
        A        = LapSLPmatrix(t,s,0)              # computation is arranged columnwised
        tau      = np.sin(np.abs(s['x'])) + np.cos(np.abs(s['x']))
        u_temp   = A.dot(tau)                       # u_temp is columnwised
        idx      = ii.reshape(ny,nx,order='F')      # uexa[idx] is rowwised
        u.T[idx.T] = u_temp.flatten()            # !!!!!!!!!!!!!!!!!!!!!
        u[~idx] = uexa[~idx]
        
        err[NN-1] = np.max(np.abs(u[idx]-uexa[idx]))
        

    fig2 = plt.figure(2)
    logerr = plt.imshow(np.log10(np.abs(np.real(u-uexa))+10**(-16)),aspect=nx/ny, interpolation='none')
    fig2.colorbar(logerr)
    plt.grid(True)
    plt.show()
    
    fig3 = plt.figure(3)
    errord = plt.semilogy(100*(np.arange(1,Nn+1)),err,'o')
    plt.show()

if __name__=='__main__':
    test_laplaceeval()
   

