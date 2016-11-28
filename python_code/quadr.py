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


# def test_setupquad():
# 	s = {}
# 	s['Z'] = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)               # starfish param
# 	n = 100;
# 	s = setupquad(s,n);
# 	s = []; s.x = Z((0:n-1)/n*2*pi); s = setupquad(s);  % testing s.x input only
# 	figure; plot(s.x,'k.'); hold on; plot([s.x, s.x+0.2*s.nx].', 'b-'); axis equal
# 	% Now check that normals from spectral differentiation are accurate:
# 	Zp = @(s) -1.5*sin(5*s).*exp(1i*s) + 1i*Z(s);        % Z' formula
# 	s.Zp = Zp;
# 	t = setupquad(s); norm(t.nx-s.nx)             % should be small
# 	s = []; s.x = 3; s.Z = Z; s = setupquad(s,100);    % N should override s.x


if __name__=='__main__':
	# test_setupquad()
	s = {}
	s['Z'] = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)
	N = 300
	s = quadr(s, N)
	# print(s)
	import matplotlib.pyplot as plt
	plt.plot(np.real(s['x']),np.imag(s['x']))
	plt.show()


