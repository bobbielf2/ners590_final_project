#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:13:02 2016

@author: haizhu
"""

import numpy as np
import scipy as sp

def LapSLPmatrix(s):
    
    N = s['x'].size
    d = np.tile(s['x'],(1,N)) - np.tile(s['x'].T,(N,1))
    temp = np.append(np.array([[0]]),s['t'][np.arange(N)])/2          # [0;s.t(1:end-1)]/2
    S = -np.log(np.abs(d)) + sp.circulant(0.5*np.log(4*np.sin(temp)**2)).T
    
    Sdim = S.shape[0]
    di = np.diag_indices(Sdim)
    S[di] = -np.log(s['sp']).T
    
    m = np.arange(1,np.int(N/2))
    rjnvec = np.append(np.append(np.array([0]), 1/m,axis=0),np.append(np.array([2/N]), 1/m[::-1],axis=0),axis=0)           # [0 1./m 2/N 1./m(end:-1:1)]
    Rjn = np.fft.ifft(rjnvec)/2
    
    S = S/N + sp.circulant(Rjn).T
    S = S*np.tile(s['sp'].T,(N,1))
    
    return S