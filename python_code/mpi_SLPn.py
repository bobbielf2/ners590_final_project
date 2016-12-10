#!/usr/bin/env python
import os, sys, time  
import numpy as np  
import scipy.io as sio
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI 

from matplotlib import path 
from quadr import quadr
from LapSLPmatrix import LapSLPmatrix

# experiment with mpi4py to parallel source to target interaction
# to run it, simply type 'mpiexec -n 2 python test_parallel_SLP.py' in terminal

# instance for invoking MPI relatedfunctions  
comm = MPI.COMM_WORLD  
# the node rank in the whole community  
comm_rank = comm.Get_rank()  
# the size of the whole community, i.e.,the total number of working nodes in the MPI cluster  
comm_size = comm.Get_size()  



# test MPI  
if __name__ == "__main__":  
    
#-----------------rank 0 processor work (read in source location, and set up target )----------------------------
   
    if comm_rank == 0: 
        
        # get all source particles 
        # load test n particle data 
        dir_path    = os.path.dirname(os.path.realpath(__file__)) + "/TestData/"
        circle      = sio.loadmat(dir_path+'circle.mat')
        all_source  = circle['x']
        N, M        = all_source.shape
       
        # set up all target points
        nx          = 100
        gx          = np.arange(1,nx+1)/nx
        ny          = 100
        gy          = np.arange(1,ny+1)/ny     # set up plotting
        xx, yy      = np.meshgrid(gx,gy)
        zz          = xx + 1j*yy
       
        t           = {}
        ii          = np.ones((nx*ny, ), dtype=bool)
        for l in range(0,M):
            s_temp      = all_source[:,l][:,np.newaxis]
            p           = path.Path(np.vstack((np.real(s_temp).T,np.imag(s_temp).T)).T)
            ii          = (~p.contains_points(np.vstack((np.real(zz).flatten('F'),np.imag(zz).flatten('F'))).T))&ii
           
        t['x']      = zz.flatten('F')[ii][np.newaxis].T
        all_target  = t['x']
       
        print ("------------- target point --------------" ) 
        print ("Process", comm_rank, "has target with size", all_target.size)

   



#----------------broadcasting source data---------------------------------------------------------
    #broadcast source to all processors  
    all_source          = comm.bcast(all_source if comm_rank == 0 else None, root = 0) 
   
    #divide source to each processor  
    num_source          = all_source.shape[1]  
    local_source_offset = np.linspace(0, num_source, comm_size + 1).astype('int') 
   
    #broadcast target to all processors
    all_target          = comm.bcast(all_target if comm_rank == 0 else None, root = 0) 
   
    
    comm.Barrier()
    #start timeing
    t_start             = MPI.Wtime()
   
   
   
   
#----------------local computation on comm_rank processor-----------------------------------------
    #get the local data which will be processed in this processor 
    #this local source and target array lives on comm_rank processor
    local_source        = all_source[:,local_source_offset[comm_rank] :local_source_offset[comm_rank + 1]]
    local_target        = all_target
    print ("------------- local target point --------------" )
    print (" %d/%d processor has local target with size %d" %(comm_rank, comm_size, local_target.size) )
 
    #get local source and target dim
    N, local_source_num = local_source.shape 
    M                   = local_target.shape[0]
    local_u             = np.zeros(M) + 1j*np.zeros(M)
    for l in range(0,local_source_num):
       
        # local source quadr structure acquire
        s_temp          = {}
        s_temp['x']     = local_source[:,l][:,np.newaxis]
        s_temp          = quadr(s_temp,N)
        # local target structure
        t_temp          = {}
        t_temp['x']     = local_target
       
        # quadrature matrix
        A               = LapSLPmatrix(t_temp,s_temp,0)
        # artificial density
        tau             = np.sin(2*np.pi*np.real(s_temp['x'])) + np.cos(np.pi*np.imag(s_temp['x']))
        u_temp          = A.dot(tau)
       
        # update local velocity at local target points
        local_u         = local_u + u_temp.flatten()
    
    print ("------------- local source point and local u --------------" )
    print (" %d/%d processor gets local source with size %d, local u with size %d" %(comm_rank, comm_size, local_source.shape[1], local_u.size) ) 

    comm.Barrier()
   
   

#------------------rank 0 processor work (post process parallel results )-----------------------------
    # the 'u' array will hold the sum of each 'u_local' array
    if comm.rank == 0:
        # only processor 0 will actually get the data
        utotal    = np.zeros_like(local_u)
    else:
        utotal    = None

    # use MPI to get the total velocity
    comm.Reduce(
         [local_u , MPI.DOUBLE],
         [utotal  , MPI.DOUBLE],
          op      = MPI.SUM,
          root    = 0
         )

    # print out the 'u'
    # only processor 0 actually has the data
    #print ('[%i]'%comm.rank, u)

    comm.Barrier()
    t_diff         = MPI.Wtime() - t_start
    if comm.rank == 0: 
        print (t_diff)
        u          = 0*(1+1j)*zz
        idx        = ii.reshape(ny,nx,order='F')
        u.T[idx.T] = utotal.flatten()
        
        # plot will cause computing issue...
#        fig = plt.figure()
#        uimage = plt.imshow(np.real(u),aspect=nx/ny, interpolation='none')
#        fig.colorbar(uimage)
#        plt.grid(True)
#        plt.show()
