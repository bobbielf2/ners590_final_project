#!/usr/bin/env python3

import sys
import numpy as np
from perispecdiff import perispecdiff

def quadr(s, N = 0):
# Set up quadrature & geometry for smooth closed curve  
# two ways of calling:
#		quadr(s)
#		quadr(s, N)
	if N != 0:
		s['t'] = np.arange(0,N)[:,np.newaxis] * (2*np.pi/N)
		if 'Z' in s:
			s['x'] = s['Z'](s['t'])    # use formula, s['Z'] is a lambda function
		if N != s['x'].size:
			sys.exit("N differs from length of s['x']; that sucks!")
	elif 'x' in s:
		s['x'] = np.array(s['x'])[:, np.newaxis]          # ensure col vec
		N = s['x'].size
		s['t'] = np.arange(0,N)[:, np.newaxis] * (2*np.pi/N) # we don't know the actual params, but choose this
	else:
		sys.exit("Need to provide at least s['Z'] and N, or s['x']. Neither found!")

	if 'Zp' in s:
		s['xp'] = s['Zp'](s['t'])
	else:
		s['xp'] = perispecdiff(s['x'])

	if 'Zpp' in s:
		s['xpp'] = s['Zpp'](s['t'])
	else:
		s['xpp'] = perispecdiff(s['xp'])

	# Now local stuff that derives from x, xp, xpp at each node...
	s['sp'] = np.abs(s['xp'])
	s['tang'] = s['xp']/s['sp']
	s['nx'] = -1j*s['tang']
	s['cur'] = -np.real(np.conj(s['xpp'])*s['nx'])/s['sp']**2  # recall real(conj(a)*b) = "a dot b"
	s['w'] = (2*np.pi/N)*s['sp']
	s['cw'] = (2*np.pi/N)*s['xp']  # complex weights (incl complex speed)
	
	return s


import matplotlib.pyplot as plt

def test_quadr():

	Z = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)               # starfish param

	s = {}
	s['Z'] = Z
	n = 100
	s = quadr(s,n)
	plt.plot(np.real(s['x']),np.imag(s['x']))
	plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
	plt.axis('equal')
	plt.title("Case 1: both s['Z'] and N input")
	plt.show()
	
	s = {}
	s['x'] = Z(np.arange(0,n)/n*2*np.pi)
	s = quadr(s)  # testing s.x input only
	plt.clf()
	plt.plot(s['x'].real, s['x'].imag)
	plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
	plt.axis('equal')
	plt.title("Case 2: s['x'] input only")
	plt.show()
	# Now check that normals from spectral differentiation are accurate:
	Zp = lambda s: -1.5 * np.sin(5*s) * np.exp(1j*s) + 1j * Z(s)        # Z' formula
	s['Zp'] = Zp
	t = quadr(s)
	print('error in the normal vec: ', np.linalg.norm(t['nx']-s['nx']))      # should be small

	s = {}
	s['x'] = 3
	s['Z'] = Z
	s = quadr(s,100)    # N should override s.x
	plt.clf()
	plt.plot(s['x'].real, s['x'].imag)
	plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
	plt.axis('equal')
	plt.title("Case 3: N override s['x']")
	plt.show()


if __name__=='__main__':
	test_quadr()
	
