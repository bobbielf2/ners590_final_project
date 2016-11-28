#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 14:01:47 2016

@author: haizhu
"""

import numpy as np

def perispecdiff(f):
    # regardless of input, the output is a column array
    N = f.size
    f = f.reshape((N,1))
    if np.mod(N,2) == 0:
        vec = np.append(np.append(np.array([0]),1j*np.arange(1,N/2),axis=0),
                        np.append(np.array([0]),1j*np.arange(-N/2+1,0),axis=0),axis=0)[np.newaxis]
        g = np.fft.ifft( np.fft.fft(f.T)*vec ).T
    else:
        vec = np.append(np.append(np.array([0]),1j*np.arange(1,(N+1)/2),axis=0),
                        1j*np.arange((1-N)/2,0),axis=0)[np.newaxis]
        g = np.fft.ifft( np.fft.fft(f.T)*vec ).T
    
    return g