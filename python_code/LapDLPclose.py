'''
Laplace DLP close evaluation

@author: Bowei Wu 12/06/2016
'''

import numpy as np
from perispecdiff import perispecdiff
from cauchyBary import cauchyBary

def LapDLPclose(x,s,tau,side,der=False):
    # Step 1. Find v^- or v^+
    vb = (0+0j)*s['x']
    taup = perispecdiff(tau)
    n = vb.size
    for i in range(0,n):
        j = list(range(0,i))+list(range(i+1,n))
        vb[i] = np.sum(np.multiply(np.divide(tau[j]-tau[i],s['x'][j]-s['x'][i]),s['cw'][j])) + taup[i]*s['w'][i]/s['sp'][i]
    vb = -1/2j/np.pi * vb
    if side == 'i':
        vb = vb - tau

    # Step 2. Cauchy integral barycentric eval
    if der:
        v, vp = cauchyBary(x,s,vb,side,der)
        u = np.real(v)
        ux = np.real(vp)
        uy = -np.imag(vp)
        return u, ux, uy
    else:
        v = cauchyBary(x,s,vb,side)
        u = np.real(v)
        return u

def testLapDLP():
    from quadr import quadr
    import matplotlib.pyplot as plt
    from LapDLPmatrix import LapDLPmatrix
    # from mpl_toolkits.mplot3d import Axes3D

    side = 'e'
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

    ub = np.real(f(s['x']))
    # upb = fp(s['x'])
    # uxb = np.real(upb)
    # uyb = -np.imag(upb)

    # target
    nx       = 150
    gx       = (np.arange(1,nx+1)/nx*2-1)*1.5
    ny       = 150
    gy       = (np.arange(1,ny+1)/ny*2-1)*1.5     # set up plotting
    xx, yy   = np.meshgrid(gx,gy)
    zz       = xx + 1j*yy

    t = {}
    if side == 'e':
        ii = s['outside'](zz)
    else:
        ii = s['inside'](zz)
    t['x'] = zz[ii,np.newaxis]

    # generate exact solution
    uexa = np.empty(zz.shape)
    uexa.fill(np.nan)
    uexa[ii,np.newaxis] = np.real(f(t['x']))
    # uxexa = 0*zz
    # uxexa[ii,np.newaxis] = np.real(fp(t['x']))

    # plot exact solution 
    Z = uexa
    plt.subplot(221)
    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=plt.cm.jet,
    #                    linewidth=0, antialiased=False)
    # ax.set_zlim(Z.max(), Z.min())
    # fig.colorbar(surf)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=np.nanmin(Z))
    plt.colorbar()
    plt.title('u exact')
    # plt.show()

    # DLP solve
    if side=='i':
        A = -np.eye(n)/2 + LapDLPmatrix(s,s)  # full rank
    else:
        A = np.eye(n)/2 + LapDLPmatrix(s,s)   # has rank-1 nullspace, ok for dense solve
    # tau = np.array(np.asmatrix(A).I.dot(ub))
    tau = np.linalg.lstsq(A,ub)[0]
    # print('A = ', A)
    # print('tau = ',tau)
    # print('A*tau - ub = ',A.dot(tau)-ub)
    # return

    # DLP close eval
    u = np.empty(zz.shape)
    u.fill(np.nan)
    # up = np.zeros(zz.shape)
    u[ii,np.newaxis] = LapDLPclose(t['x'],s,tau,side)
    # u[ii,np.newaxis], up[ii,np.newaxis] = LapDLPclose(t['x'],s,ub,side,der=1)

    # plot barycentric solution 
    plt.subplot(223)
    Z = u
    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=plt.cm.jet,
    #                    linewidth=0, antialiased=False)
    # ax.set_zlim(Z.max(), Z.min())
    # fig.colorbar(surf)
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=np.nanmin(Z))
    plt.colorbar()
    plt.title('u approx')
    # plt.show()

    # plot error in u
    plt.subplot(224)
    Z = np.log10(abs(u-uexa))
    plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
                origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
                vmax=np.nanmax(Z), vmin=Z[Z > np.NINF].min())
    plt.colorbar()
    plt.title('u log10 error')
    plt.show()

    # # plot error in ux
    # Z = np.log10(abs(ux-uxexa))
    # plt.imshow(Z, interpolation='nearest', cmap=plt.cm.jet,
    #             origin='lower', extent=[-1.5, 1.5, -1.5, 1.5],
    #             vmax=np.nanmax(Z), vmin=Z[Z > np.NINF].min())
    # plt.colorbar()
    # plt.title('ux log10 error')
    # plt.show()
    
    # print(Z.min(),Z[Z > -np.inf].min())
    print('u max error: ', np.max(np.abs(u[ii]-uexa[ii])))
    # print('up max error: ', np.max(np.abs(up[ii]-upexa[ii])))



if __name__ == '__main__':
    testLapDLP()
