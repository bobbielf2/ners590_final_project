#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:28:10 2016

@author: haizhu
"""

import numpy as np

def LapSLPmatrix(t,s,a):
    
    N = s['x'].size
    M = t['x'].size
    d = np.tile(t['x'],(1,N)) - np.tile(s['x'].T+a,(M,1))
    ny = np.tile(s['nx'].T,(M,1))
    A = -1/2/np.pi*np.log(np.abs(d))*np.tile(s['w'].T,(M,1))
    
    nx = np.tile(-t['nx'],(1,N))
    An = 1/2/np.pi*np.real(nx/d)
    if (np.shape(s['x'])==np.shape(t['x'])) & (np.max(np.abs(s['x']+a-t['x'])))<np.exp(-14):
        Andim = An.shape[0]
        di = np.diag_indices(Andim)
        An[di] = -s['cur']/4/np.pi
    An = An*np.tile(s['w'].T,(M,1))
    
    
    return A, An