#!/usr/bin/env python3
"""
Created on Mon Nov 28 11:00:33 2016

@author: haizhu, Bobbie Wu
"""
import numpy as np
def LapDLPmatrix(t,s,a=0,der=0):
    
    N = s['x'].size
    M = t['x'].size
    d = np.tile(t['x'],(1,N)) - np.tile(s['x'].T+a,(M,1))
    ny = np.tile(s['nx'].T,(M,1))
    A = 1/2/np.pi*np.real(ny/d)
    
    if M == N and (np.max(np.abs(s['x']+a-t['x'])))<1e-14:
        # print('computing diag elements...',s['cur'].shape)
        Adim = A.shape[0]
        di = np.diag_indices(Adim)
        A[di] = -s['cur'].flatten()/4/np.pi
    A = np.asarray(A)*s['w'].T
    
    if der:
        if 'nx' not in t:
            raise Exception("target normal t['nx'] not exist!")
        csry = np.conjugate(ny)*d
        nx = np.tile(t['nx'],(1,N))
        csrx = np.conjugate(nx)*d
        r = np.abs(d)
        An = -np.real(csry*csrx)/(r**4)/(2*np.pi)
        An = An*np.tile(s['w'].T,(M,1))
        return A, An

    return A

#        t1 = np.real(s['tang'])
#        t2 = np.imag(s['tang'])
#        sdim = s['x'].shape
#        di = np.diag_indices(2*sdim)
#        A[di] = -np.append(s['cur']*t1**2,s['cur']*t2**2,axis=0)/2/np.pi
#        A = A[:,np.append(np.arange(sdim,2*sdim),np.arange(sdim),axis=0)]
    
    
#    r2 = d*np.conjugate(d)
#    
#    dot_part = np.real(ny/d)/r2/np.pi
#    dot_part = np.tile(dot_part,(2,2))
#    
#    d1 = np.real(d)
#    d2 = np.imag(d)
#    cross_part = np.append(np.append(d1**2,d1*d2,axis=1),np.append(d1*d2,d2**2,axis=1),axis=0)
#    A = dot_part*cross_part
    
 
              
      #  A[di] = A[:,np.append(np.arange(3,6),np.arange(3),axis=1)]
        
    