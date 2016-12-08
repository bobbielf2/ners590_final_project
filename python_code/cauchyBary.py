'''
Evaluate Cauchy integral using barycentric formula

@author: Bowei Wu 12/06/2016
'''

import numpy as np

def cauchyBary(x,s,vb,side,der=0):
    N = x.size
    denom = np.array(np.tile(s['cw'].T,(N,1))/(s['x'].T-x))
    numer = np.sum(vb.T*denom, 1, keepdims=True)
    denom = np.sum(denom, 1, keepdims=True)
    if side == 'e':
        denom = denom - 2j*np.pi
    v = np.asarray(numer)/np.asarray(denom)

    if der:
        vpb = 0*vb
        n = vb.size
        for i in range(n):
            j = list(range(i))+list(range(i+1,n))
            vpb[i] = np.sum(np.asarray(vb[j]-vb[i])/np.asarray(s['x'][j]-s['x'][i])*np.asarray(s['cw'][j]))
        if side == 'e':
            vpb = vpb + 2j*np.pi*vb
        vpb = vpb * (-1/s['cw'])
        vp = cauchyBary(x,s,vpb,side)
        return v, vp

    return v

def testCauchy():
    from quadr import quadr
    import matplotlib.pyplot as plt

    side = 'i'
    s = {}
    s['Z'] = lambda t: (1 + 0.3 * np.cos(5 * t)) * np.exp(1j*t)     # starfish param
    # s['Zp'] = lambda t: (Rp(t) + 1j*R(t)) * np.exp(1j*t)
    # s['Zpp'] = lambda t: (Rpp(t) + 2i*Rp(t) - R(t)) * np.exp(1j*t)
    R = lambda t: 1 + 0.3 * np.cos(5 * t)
    s['inside'] = lambda z: np.abs(z) < R(np.angle(z))
    s['outside'] = lambda z: np.abs(z) > R(np.angle(z))
    n = 200
    s = quadr(s, n)
    # plt.clf()
    # plt.plot(s['x'].real, s['x'].imag)
    # plt.quiver(s['x'].real, s['x'].imag, s['nx'].real, s['nx'].imag) # plot unit normals
    # plt.axis('equal')
    # plt.show()

    # construct anal soln
    a = 1.1+1j
    if side == 'e':
        a = 0.1+0.3j
    f = lambda z: 1/(z-a)
    fp = lambda z: -1/(z-a)**2

    ub = f(s['x'])
    ubp = fp(s['x'])

    # target
    nx       = 150
    gx       = (np.arange(1,nx+1)/nx*2-1)*1.5
    ny       = 150
    gy       = (np.arange(1,ny+1)/ny*2-1)*1.5     # set up plotting
    xx, yy   = np.meshgrid(gx,gy)
    zz       = xx + 1j*yy

    t        = {}
    if side == 'e':
        ii = s['outside'](zz)
    else:
        ii = s['inside'](zz)
    t['x']   = zz[ii,np.newaxis]

    # generate exact solution
    uexa = np.nan*zz
    uexa[ii,np.newaxis] = f(t['x'])
    upexa = np.nan*zz
    upexa[ii,np.newaxis] = fp(t['x'])

    # plot exact solution 
    Z = np.real(uexa)
    plt.subplot(221)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=np.nanmin(Z))
    plt.title('exact')
    plt.colorbar()
    # plt.show()

    # compute cauchyBary
    u = np.nan*zz
    up = np.nan*zz
    # u[ii,np.newaxis] = cauchyBary(t['x'],s,ub,side)
    u[ii,np.newaxis], up[ii,np.newaxis] = cauchyBary(t['x'],s,ub,side,der=1)

    # plot barycentric solution 
    Z = np.real(u)
    plt.subplot(222)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=np.nanmin(Z))
    plt.colorbar()
    plt.title('approx')
    # plt.show()

    # plot error in u
    Z = np.log10(abs(u-uexa))
    print(-np.inf > np.NINF)
    plt.subplot(223)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=Z[Z > np.NINF].min())
    plt.colorbar()
    plt.title('u log10 error')
    # plt.show()

    # plot error in up
    Z = np.log10(abs(up-upexa))
    plt.subplot(224)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=Z[Z > np.NINF].min())
    plt.colorbar()
    plt.title('up log10 error')
    plt.show()
    
    # print(Z.min(),Z[Z > -np.inf].min())
    print('u max error: ', np.max(np.abs(u[ii]-uexa[ii])))
    print('up max error: ', np.max(np.abs(up[ii]-upexa[ii])))



if __name__ == '__main__':
    testCauchy()

