"""
m. sisti, f. finelli - 18/11/2020
e-mail: manuela.sisti@univ-amu.fr, francesco.finelli@phd.unipi.it

The goal of this routine is to create quantities that can be "correlated" to obtain regions interesting for reconnection. 
Basic fields (J,B,Ve,n,E) must be loaded in the format [3,nx,ny,nz], where nx,ny and nz are the grid dimensions.
--------------------------------------------------------------------------
The final quantities computed by the routine are:
--> the magnitude of the current density 								CALLED Jn
--> |ve_plane| 												CALLED mod_Ve_plane
--> the magnitude of the z-component of the electric field 					 	CALLED aEdec
--> the magnitude of the curl of the electron velocity							CALLED Omega_ve
--> the magnitude of the in-plane magnetic field 							CALLED aB_inplane
--> the energy conversion term 										CALLED JE

"""

import numpy as np
import utilities_unsup as ut

#============================================================================
#calculating Jn
Jn = np.sqrt(J[0,:,:,:]**2+J[1,:,:,:]**2+J[2,:,:,:]**2)

#============================================================================
#calculating mod_Ve_plane

mod_Ve_plane = np.sqrt(Ve[0,:,:,:]**2+Ve[1,:,:,:]**2)

#============================================================================
#calculating aEdec

Edec = np.zeros((3,nx,ny,nz))
crossVeB = np.zeros((3,nx,ny,nz))
crossVeB[0,:,:,:], crossVeB[1,:,:,:], crossVeB[2,:,:,:] = ut.calc_cross(ut,Ve[0,:,:,:],B[0,:,:,:],Ve[1,:,:,:],B[1,:,:,:],Ve[2,:,:,:],B[2,:,:,:])
Edec[0,:,:,:] = E[0,:,:,:] + crossVeB[0,:,:,:]
Edec[1,:,:,:] = E[1,:,:,:] + crossVeB[1,:,:,:]
Edec[2,:,:,:] = E[2,:,:,:] + crossVeB[2,:,:,:]

aEdec = np.absolute(Edec[2,:,:,:])

#============================================================================
#calculating Omega_ve
Omega_ve_x = ut.calc_grady(Ve[2,:,:,:],nx,ny,nz,dl) - ut.calc_gradz(Ve[1,:,:,:],nx,ny,nz,dl)
Omega_ve_y = ut.calc_gradz(Ve[0,:,:,:],nx,ny,nz,dl) - ut.calc_gradx(Ve[2,:,:,:],nx,ny,nz,dl)
Omega_ve_z = ut.calc_gradx(Ve[1,:,:,:],nx,ny,nz,dl) - ut.calc_grady(Ve[0,:,:,:],nx,ny,nz,dl)
Omega_ve = np.sqrt(Omega_ve_x**2+Omega_ve_y**2+Omega_ve_z**2)


#============================================================================
#calculating aB_inplane
aB_inplane = np.sqrt(B[0,:,:,0]**2+B[1,:,:,0]**2)


#============================================================================
#calculating JE
JE = np.zeros((nx,ny,nz))
JE = ut.calc_scalr(ut,J[0,...],Edec[0,...],J[1,...],Edec[1,...],J[2,...],Edec[2,...])

#----------------------------------------------------------------------------
#ALTERNATIVE, valid for HVM code
'''
#calculating grad(P)/n in order to subtract it from the electric field 
#dx(Pe)/n=dx(nTe)/n=Te*dx(n)/n where Te=1.0*Ti=1.0*mi*beta/2=1/2=0.5
dxP = 0.5*ut.calc_gradx(N[:,:,:],nx,ny,nz,dl)
dyP = 0.5*ut.calc_grady(N[:,:,:],nx,ny,nz,dl)
dzP = 0.5*ut.calc_gradz(N[:,:,:],nx,ny,nz,dl)

#calculating JE without the contribution of the pressure term, i.e. J*(Edec+grad(P))
JE = np.zeros((nx,ny,nz))
Edec_nP = np.zeros((2,nx,ny,nz))
Edec_nP[0,:,:,:] = Edec[0,:,:,:] + dxP/N
Edec_nP[1,:,:,:] = Edec[1,:,:,:] + dyP/N
JE = ut.calc_scalr(ut,J[0,...],Edec_nP[0,...],J[1,...],Edec_nP[1,...],J[2,...],Edec[2,...])
'''

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

#saving...Warning: you have to change the path
fname = path_alldata

file1 = open(fname, 'w')
for i in range(0,nx):
    for j in range(0,ny):
        tw1 = '%.7f' % Jn[i,j,0]
        tw2 = '%d' % i
        tw3 = '%d' % j
        tw4 = '%.7f' % mod_Ve_plane[i,j,0]
        tw5 = '%.7f' % Omega_ve[i,j,0]
        tw6 = '%.7f' % aEdec[i,j,0]
        tw7 = '%.7f' % aB_inplane[i,j]
        tw8 = '%.7f' % JE[i,j,0]

        #all quantities
        file1.write(tw1 + '\t' + tw2 + '\t' + tw3 + '\t' + tw4 + '\t' + tw5 + '\t' + tw6 + '\t' + tw7 + '\t' + tw8 + '\n')
file1.close()

'''
alld = np.zeros((nx*ny,8))

file2 = open(fname, 'r')
for i in range(0,nx*ny):
    tr = file2.readline().split()
    alld[i,0] = float(tr[0])
    alld[i,1] = float(tr[1])
    alld[i,2] = float(tr[2])
    alld[i,3] = float(tr[3])
    alld[i,4] = float(tr[4])
    alld[i,5] = float(tr[5])
    alld[i,6] = float(tr[6])
    alld[i,7] = float(tr[7])
file2.close()
'''



