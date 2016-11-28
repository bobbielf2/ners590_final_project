#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:27:53 2016

@author: haizhu
"""

import numpy as np
# from numpy import matlib

def quadr(s, N, qtype, qntype):
    
    if qtype == 'g':
        t        = np.arange(1,N+1)[:, np.newaxis]/N*2*np.pi # create vertical array in parameter space
        s['tlo'] = 0
        s['thi'] = 2*np.pi
        s['p']   = N
        s['w']   = 2*np.pi*np.ones((N,1))/N
        s['xlo'] = s['Z'](s['tlo'])                             # lambda function s['Z'] = lambda x: x**2 + 4
        s['xhi'] = s['Z'](s['thi'])
        npanel       = 1
        if qntype == 'G':
            _, _, D = gauss(N)                                        # need to take care of
        else:
            _, _, D = cheby(N)
            
    elif qtype == 'p':
        if 'p' in s:
            p = s['p']
        else:
            s['p'] = 16
        npanel   = np.ceil(N/p)
        N        = p*npanel
        s['tlo'] = np.arange(0,npanel)[:, np.newaxis]/npanel*2*np.pi
        s['xlo'] = s['Z'](s['tlo'])
        s['thi'] = np.arange(1,npanel+1)[:, np.newaxis]/npanel*2*np.pi
        s['xhi'] = s['Z'](s['thi'])
        pt       = 2*np.pi/npanel;
        t        = np.zeros((N,1))
        s['w']   = t
                     
        if qntype == 'G':
            x, w, D = gauss(p)
        else:
            x, w, D = cheby(p)
        D = D*2/pt
        
        for i in range(1,npanel+1):
            ii           = (i-1)*p + np.arange(1,p+1)
            t[ii-1]      = s['tlo'][i-1] + (1+x)/2*pt               # it is valid
            s['w'][ii-1] = w*pt/2
            
    s['x']   = s['Z'](t)
    s['xp']  = np.zeros(np.shape(s['x']))
    s['xpp'] = np.zeros(np.shape(s['x']))
    
    if 'Zp' in s:
        s['xp'] = s['Zp'](t)
    elif qtype == 'p':
        for i in range(1,npanel+1):
            ii            = (i-1)*p + np.arange(1,p+1)
            s['xp'][ii-1] = D.dot(s['x'][ii-1])
    else:
        s['xp'] = D.dot(s['x'])
        
    if 'Zpp' in s:
        s['xpp'] = s['Zpp'](t)
    elif qtype == 'p':
        for i in range(1,npanel+1):
            ii             = (i-1)*p + np.arange(1,p+1)
            s['xpp'][ii-1] = D.dot(s['xp'][ii-1])
    else:
        s['xpp'] = D.dot(s['xp'])
        
    s['sp']   = np.abs(s['xp'])
    s['tang'] = s['xp']/s['sp']
    s['nx']   = -1j*s['tang']
    s['cur']  = -np.real(np.conjugate(s['xpp'])*s['nx'])/(s['sp']*s['sp'])
    s['ws']   = s['w']*s['sp']
    s['t']    = t
    s['wxp']  = s['w']*s['xp']
    
    return s, N, npanel
    
 
def cheby(N):
    theta = np.pi*(2*np.arange(1,N+1)[:, np.newaxis]-1)/(2*N)
    x     = -np.cos(theta)
    
    l     = np.int(np.floor(N/2)+1)
    K     = np.arange(0,N-l+1)
    v     = np.append(2*np.exp(1j*np.pi*K/N)/(1-4*K**2),np.zeros((1,l)))
    w     = np.real(np.fft.ifft(v[:N] + np.conjugate(v[N:0:-1])))[np.newaxis].T
    
    X     = np.tile(x,(1,N))
    dX    = X - X.T
    a     = np.prod(dX+np.eye(N),axis=1)[np.newaxis]
    D     = (a.T/np.conjugate(a))/(dX+np.eye(N))
    D     = D - np.diag(np.sum(D,axis=1))
    
    return x, w, D
    
def DLPmatrix(t,s):
    
    N = s['x'].size
    M = t['x'].size
    d = np.tile(t['x'],(1,N)) - np.tile(s['x'].T,(M,1))
    ny = np.tile(s['nx'].T,(M,1))
    r2 = d*np.conjugate(d)
    
    dot_part = np.real(ny/d)/r2/np.pi
    dot_part = np.tile(dot_part,(2,2))
    
    d1 = np.real(d)
    d2 = np.imag(d)
    cross_part = np.append(np.append(d1**2,d1*d2,axis=1),np.append(d1*d2,d2**2,axis=1),axis=0)
    A = dot_part*cross_part
    
    if (np.shape(s['x'])==np.shape(t['x'])) & (np.max(np.abs(s['x']-t['x'])))<np.exp(-14):
        t1 = np.real(s['tang'])
        t2 = np.imag(s['tang'])
        sdim = s['x'].shape
        di = np.diag_indices(2*sdim)
        A[di] = -np.append(s['cur']*t1**2,s['cur']*t2**2,axis=0)/2/np.pi
        A = A[:,np.append(np.arange(sdim,2*sdim),np.arange(sdim),axis=0)]
              
      #  A[di] = A[:,np.append(np.arange(3,6),np.arange(3),axis=1)]
        
    
    return r2
    
#def diagind(A):
#    N = A.shape[0]
#    i = ravel_multi_index
#    return i
  
#%%    
def test_stokesSD():
    
    # source
    epsilon  = 0.1
    a        = 4.0
    b        = 1.5
    w        = np.pi/12
    U        = lambda t: 2*np.pi - t + 1j*(3/8*w*np.tanh(epsilon*((np.pi-t)**a-(np.pi/b)**a))+7/8*w)
    D        = lambda t: t - 1j*(3/8*w*np.tanh(epsilon*((t-np.pi)**a-(np.pi/b)**a))+7/8*w)
    
    su       = {}
    su['Z']  = U
    sd       = {}
    sd['Z']  = D
    
    su, N, npanel = quadr(su, 10, 'g', 'C')
    
    # target
    nx       = 30
    gx       = np.arange(1,nx+1)/nx*2*np.pi
    ny       = 15
    gy       = np.arange(1,ny+1)/ny*2-1
    xx, yy   = np.meshgrid(gx,gy)
    zz       = xx + 1j*yy
    inside   = lambda z: (np.imag(z)<np.imag(U(np.real(z))))*1 & (np.imag(z)>np.imag(D(np.real(z))))*1  # turn boolean arrary to 0 1
    ii       = inside(zz)
    t        = {}
    t['x']   = zz[ii==1][np.newaxis].T    # bool operation (should work, there might be other way)
    
    
    # self convergence test
    N        = 10
    f        = np.nan*zz
    lptype   = 's'
    tau      = {}

    # return s
    


if __name__=='__main__':
#    from matplotlib import pyplot as plt
    x, w, D = cheby(11)
    test_stokesSD()
    
    
    
    
