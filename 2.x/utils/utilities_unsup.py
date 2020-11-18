"""
Some utilities
"""

from scipy.spatial.distance import pdist
import scipy.ndimage as ndm
import scipy.signal as scip
import scipy.interpolate as itp
import scipy.spatial as sp
from scipy.spatial import distance
from scipy.stats import pearsonr 

import numpy as np
from numpy import linalg as LA

#==========================================================

def calc_gradx(tar_var,
        nx,
        ny,
        nz,
        dl, 
        periodic = True,
        der_ord = 1):
        """ 
        ------------------------------------------------------------------------------------
        calculates x gradient component with(out) periodical boundary conditions
        ------------------------------------------------------------------------------------
        tar_var           target variable of the procedure, if is a vector you have to give the single component
        periodic = True   [bool] periodic boundary conditions?
        der_ord = 1       [int] order of derivation (0: no derivative, 1: first derivative ...)
        ------------------------------------------------------------------------------------
        """

        #determine the coefficients
        if nx == 1 : 
            return np.zeros([nx,ny,nz])

        else : 
            oo_6 = dl[0]**(-der_ord) * np.array([1.0,  0.0])        
            aa_6 = dl[0]**(-der_ord) * np.array([0.0,  9.0/12.0]) 
            bb_6 = dl[0]**(-der_ord) * np.array([0.0, -3.0/20.0]) 
            cc_6 = dl[0]**(-der_ord) * np.array([0.0,  1.0/60.0]) 
    
            #create the new vector, fill it
            if periodic :
                ff = np.tile(tar_var,(2,1,1))
                dx_f  = oo_6[der_ord] * ff[0:nx,:,:] 
                dx_f += aa_6[der_ord] *(ff[1:1+nx,:,:] - ff[nx-1:2*nx-1,:,:])
                dx_f += bb_6[der_ord] *(ff[2:2+nx,:,:] - ff[nx-2:2*nx-2,:,:])
                dx_f += cc_6[der_ord] *(ff[3:3+nx,:,:] - ff[nx-3:2*nx-3,:,:])
            else :
                f = tar_var  
                dx_f  = np.zeros([nx,ny,nz])  
                dx_f[3:nx-3,:,:]  = oo_6[der_ord] * f[3:nx-3,:,:]
                dx_f[3:nx-3,:,:] += aa_6[der_ord] *(f[4:nx-2,:,:] - f[2:nx-4,:,:])
                dx_f[3:nx-3,:,:] += bb_6[der_ord] *(f[5:nx-1,:,:] - f[1:nx-5,:,:])
                dx_f[3:nx-3,:,:] += cc_6[der_ord] *(f[6:nx-0,:,:] - f[0:nx-6,:,:])

           
        return dx_f

#==========================================================

def calc_grady(tar_var,
        nx,
        ny,
        nz,
        dl,
        periodic = True,
        der_ord = 1): 
        """ 
        ------------------------------------------------------------------------------------
          calculates y gradient component with(out) periodical boundary conditions
        ------------------------------------------------------------------------------------
        tar_var            [str OR np.ndarray] target variable of the procedure, if is a vector you have to give the single component
        periodic = True   [bool] periodic boundary conditions?
        der_ord = 1       [int] order of derivation (0: no derivative, 1: first derivative ...)
        ------------------------------------------------------------------------------------
        """

        #determine the coefficients
        
        if ny == 1 : 
            return np.zeros([nx,ny,nz])

        else :
            oo_6 = dl[1]**(-der_ord) * np.array([1.0,  0.0])       
            aa_6 = dl[1]**(-der_ord) * np.array([0.0,  9.0/12.0])
            bb_6 = dl[1]**(-der_ord) * np.array([0.0, -3.0/20.0])
            cc_6 = dl[1]**(-der_ord) * np.array([0.0,  1.0/60.0])
      
            #create the new vector, fill it  (periodic / nonperiodic boundary cases ...)  
            if periodic :
                ff = np.tile(tar_var,(1,2,1))
                dy_f  = oo_6[der_ord] * ff[:,0:ny,:] 
                dy_f += aa_6[der_ord] *(ff[:,1:1+ny,:] - ff[:,ny-1:2*ny-1,:])
                dy_f += bb_6[der_ord] *(ff[:,2:2+ny,:] - ff[:,ny-2:2*ny-2,:])
                dy_f += cc_6[der_ord] *(ff[:,3:3+ny,:] - ff[:,ny-3:2*ny-3,:])
            else :
                f = tar_var
                dy_f = np.zeros([nx,ny,nz])  
                dy_f[:,3:ny-3,:]  = oo_6[der_ord] * f[:,3:ny-3,:]
                dy_f[:,3:ny-3,:] += aa_6[der_ord] *(f[:,4:ny-2,:] - f[:,2:ny-4,:])
                dy_f[:,3:ny-3,:] += bb_6[der_ord] *(f[:,5:ny-1,:] - f[:,1:ny-5,:])
                dy_f[:,3:ny-3,:] += cc_6[der_ord] *(f[:,6:ny-0,:] - f[:,0:ny-6,:])

        
        return dy_f

#==========================================================

def calc_gradz(tar_var,
        nx,
        ny,
        nz,
        dl,
        periodic = True,
        der_ord = 1):
        """ 
        ------------------------------------------------------------------------------------    
          calculates z gradient component with(out) periodical boundary conditions
        ------------------------------------------------------------------------------------
        tar_var             [str OR np.ndarray] target variable of the procedure, if is a vector you have to give the single component
        periodic = True    [bool] periodic boundary conditions?
        der_ord = 1        [int] order of derivation (0: no derivative, 1: first derivative ...)
        ------------------------------------------------------------------------------------
        """

        #determine the coefficients

        if nz == 1 :
            return np.zeros([nx,ny,nz])

        else :
            oo_6 = dl[2]**(-der_ord) * np.array([1.0,  0.0])      
            aa_6 = dl[2]**(-der_ord) * np.array([0.0,  9.0/12.0]) 
            bb_6 = dl[2]**(-der_ord) * np.array([0.0, -3.0/20.0]) 
            cc_6 = dl[2]**(-der_ord) * np.array([0.0,  1.0/60.0]) 

            #create the new vector, fill it
            if periodic :
                ff = np.tile(tar_var,(1,1,2))
                dz_f  = oo_6[der_ord] * ff[:,:,0:nz] 
                dz_f += aa_6[der_ord] *(ff[:,:,1:1+nz] - ff[:,:,nz-1:2*nz-1])
                dz_f += bb_6[der_ord] *(ff[:,:,2:2+nz] - ff[:,:,nz-2:2*nz-2])
                dz_f += cc_6[der_ord] *(ff[:,:,3:3+nz] - ff[:,:,nz-3:2*nz-3])  
            else:
                f = tar_var 
                dz_f  = np.zeros([nx,ny,nz])  
                dz_f[:,:,3:nz-3]  = oo_6[der_ord] * f[:,:,3:nz-3]
                dz_f[:,:,3:nz-3] += aa_6[der_ord] *(f[:,:,4:nz-2] - f[:,:,2:nz-4])
                dz_f[:,:,3:nz-3] += bb_6[der_ord] *(f[:,:,5:nz-1] - f[:,:,1:nz-5])
                dz_f[:,:,3:nz-3] += cc_6[der_ord] *(f[:,:,6:nz-0] - f[:,:,0:nz-6])

        
        return dz_f


#------------------------------------------------------------
def calc_scalr(self,  
        tar_var_1x,  
        tar_var_2x,  
        tar_var_1y,  
        tar_var_2y,
        tar_var_1z = None,
        tar_var_2z = None):  
        """
        -------------------------------------------------------------------------------------
        calculates scalar product between two vectors
        -------------------------------------------------------------------------------------
        tar_var_1x          [var]  first target variable of the procedure: x component
        tar_var_2x          [var]  second target variable of the procedure: x component
        tar_var_1y          [var]  first target variable of the procedure: y component
        tar_var_2y          [var]  second target variable of the procedure: y component
        tar_var_1z = None   [None OR var]  first target variable of the procedure: z component
        tar_var_2z = None   [None OR var]  second target variable of the procedure: z component
        -------------------------------------------------------------------------------------
        new_var             [var]
        -------------------------------------------------------------------------------------
        """

        tar_arr  = np.multiply(tar_var_1x,tar_var_2x)
        tar_arr += np.multiply(tar_var_1y,tar_var_2y)
        if not (tar_var_1z is None) :
          tar_arr += np.multiply(tar_var_1z,tar_var_2z)

        return tar_arr

#------------------------------------------------------------
def calc_cross(self,  
        tar_var_1x,
        tar_var_2x,
        tar_var_1y,  
        tar_var_2y,
        tar_var_1z = None,  
        tar_var_2z = None):  
        """
        -------------------------------------------------------------------------------------
        calculates cross product between two vectors
        -------------------------------------------------------------------------------------
        tar_var_1x          [var]  first target variable of the procedure: x component
        tar_var_2x          [var]  second target variable of the procedure: x component
        tar_var_1y          [var]  first target variable of the procedure: y component
        tar_var_2y          [var]  second target variable of the procedure: y component
        tar_var_1z = None   [None OR var]  first target variable of the procedure: z component
        tar_var_2z = None   [None OR var]  second target variable of the procedure: z component
        -------------------------------------------------------------------------------------
        new_var             [var]
        -------------------------------------------------------------------------------------
        """

        if not (tar_var_1z is None) :
          tar_arr_x  = np.multiply(tar_var_1y,tar_var_2z)
          tar_arr_x -= np.multiply(tar_var_1z,tar_var_2y)
          tar_arr_y  = np.multiply(tar_var_1z,tar_var_2x)
          tar_arr_y -= np.multiply(tar_var_1x,tar_var_2z)
        tar_arr_z  = np.multiply(tar_var_1x,tar_var_2y)
        tar_arr_z -= np.multiply(tar_var_1y,tar_var_2x)

        if not (tar_var_1z is None) : 
          return tar_arr_x, tar_arr_y, tar_arr_z
        else : 
          return tar_arr_z

#---------------------------------------------------------
def Hmatrix(self, 
        AA, 
        tot_num, 
        coordx, 
        coordy, 
        nx,
        ny,
        nz,
        dl,
        normalization = True): 
        """
        -------------------------------------------------------------------------------------
        calculates the Hessian matrix of a scalar field in a given list of points
        -------------------------------------------------------------------------------------
        AA	            [var]  scalar field in the form AA[nx,ny,nz]
        tot_num             [int]  total number of the points over which you want to compute the H matrix
        coordx              [array]  x-coordinates of the points over which you want to compute the H matrix
        coordy              [array]  x-coordinates of the points over which you want to compute the H matrix
        nx                  [int]  box first dimension
        ny                  [int]  box second dimension
        nz                  [int]  box third dimension
        normalization       [bool]  choose if you want to normalize the Hessian matrix value to the local values of the scalar field AA
        -------------------------------------------------------------------------------------
        new_var             [array]  Hessian matrix, eigenvalues and eigenvectors
        -------------------------------------------------------------------------------------
        """

        dxAA = np.zeros((nx,ny,nz))
        dyAA = np.zeros((nx,ny,nz))
        dzAA = np.zeros((nx,ny,nz))

        dxAA = self.calc_gradx(AA[:,:,:],nx,ny,nz,dl)
        dyAA = self.calc_grady(AA[:,:,:],nx,ny,nz,dl)
        dzAA = self.calc_gradz(AA[:,:,:],nx,ny,nz,dl)

        dxdxAA = np.zeros((nx,ny,nz))
        dydyAA = np.zeros((nx,ny,nz))
        dzdzAA = np.zeros((nx,ny,nz))

        dxdxAA = self.calc_gradx(dxAA[:,:,:],nx,ny,nz,dl)
        dydyAA = self.calc_grady(dyAA[:,:,:],nx,ny,nz,dl)
        dzdzAA = self.calc_gradz(dzAA[:,:,:],nx,ny,nz,dl)

        dxdyAA = np.zeros((nx,ny,nz))
        dxdzAA = np.zeros((nx,ny,nz))
        dydzAA = np.zeros((nx,ny,nz))

        dxdyAA = self.calc_gradx(dyAA[:,:,:],nx,ny,nz,dl)
        dxdzAA = self.calc_gradx(dzAA[:,:,:],nx,ny,nz,dl)
        dydzAA = self.calc_grady(dzAA[:,:,:],nx,ny,nz,dl)

        #simmetric, thus the three following are not necessary
        dydxAA = np.zeros((nx,ny,nz)) #=dxdyAA
        dzdxAA = np.zeros((nx,ny,nz)) #=dxdzAA
        dzdyAA = np.zeros((nx,ny,nz)) #=dydzAA

        dydxAA = dxdyAA
        dzdxAA = dxdzAA
        dzdyAA = dydzAA 

        H = np.zeros((3,3, tot_num))
        Jm = np.zeros(( tot_num))

        if (nz == 1):
            for i in range(0, tot_num):
                H[0,0,i] = dxdxAA[int(coordx[i]),int(coordy[i]),0]
                H[0,1,i] = dxdyAA[int(coordx[i]),int(coordy[i]),0]
                H[0,2,i] = dxdzAA[int(coordx[i]),int(coordy[i]),0]
    
                H[1,0,i] = dydxAA[int(coordx[i]),int(coordy[i]),0]
                H[1,1,i] = dydyAA[int(coordx[i]),int(coordy[i]),0]
                H[1,2,i] = dydzAA[int(coordx[i]),int(coordy[i]),0]

                H[2,0,i] = dzdxAA[int(coordx[i]),int(coordy[i]),0]
                H[2,1,i] = dzdyAA[int(coordx[i]),int(coordy[i]),0]
                H[2,2,i] = dzdzAA[int(coordx[i]),int(coordy[i]),0]

                if (normalization == True):
                    #print('Normalized at local field')
                    Jm[i] = AA[int(coordx[i]),int(coordy[i]),0]
                else:
                    #print('Non normalized at local field')
                    Jm[i] = 1.0
        else:
            for i in range(0, tot_num):
                H[0,0,i] = dxdxAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[0,1,i] = dxdyAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[0,2,i] = dxdzAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]

                H[1,0,i] = dydxAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[1,1,i] = dydyAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[1,2,i] = dydzAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]

                H[2,0,i] = dzdxAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[2,1,i] = dzdyAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                H[2,2,i] = dzdzAA[int(coordx[i]),int(coordy[i]),int(coordz[i])]

                if (normalization == True):
                    #print('Normalized at local field')
                    Jm[i] = AA[int(coordx[i]),int(coordy[i]),int(coordz[i])]
                else:
                    #print('Non normalized at local field')
                    Jm[i] = 1.0

        eigenvalues2D = np.zeros((2, tot_num))
        eigenvalues3D = np.zeros((3, tot_num))
        eigenvectors2D = np.zeros((2,2, tot_num))
        eigenvectors3D = np.zeros((3,3, tot_num))
        Hn = np.zeros((3,3, tot_num))

        for i in range(0, tot_num):
            Hn[:,:,i] = H[:,:,i]/Jm[i]    
            #computing eigenvalues and eigenvectors
            if (nz == 1):
                w, v = LA.eig(Hn[0:2,0:2,i])
                eigenvalues2D[:,i] = w
                eigenvectors2D[:,:,i] = v 
            else: 
                w, v = LA.eig(Hn[:,:,i])
                eigenvalues3D[:,i] = w
                eigenvectors3D[:,:,i] = v

        if (nz == 1):
             return Hn, eigenvalues2D, eigenvectors2D
        else:
             return Hn, eigenvalues3D, eigenvectors3D 
    
#-------------------------------------------------------------------------
def interpolation_for_width_and_thickness(self,
        eigenvalues2D, 
        eigenvectors2D,
        tot_num,
        coordx,
        coordy,
        dl,
        AA,
        nx,
        ny,
        nz):
        """
        -------------------------------------------------------------------------------------
        calculates the thickness (and the width) of a structure interpolating a scalar quantity along the directions given by the eigenvectors of the Hessian matrix and computing the full width at half maximum 
        -------------------------------------------------------------------------------------
        eigenvalues2D       [array]  eigenvalues of the Hessian matrix
        eigenvectors2D      [array]  eigenvectors of the Hessian matrix
        tot_num             [int]  total number of the points over which you want to compute the H matrix
        coordx              [array]  x-coordinates of the points over which you want to compute the H matrix
        coordy              [array]  x-coordinates of the points over which you want to compute the H matrix
        dl                  [float,float,float]  box resolution 
        AA              [array]  scalar field, doubled along each direction using np.tile
        nx                  [int]  box first dimension
        ny                  [int]  box second dimension
        nz                  [int]  box third dimension
        -------------------------------------------------------------------------------------
        new_var             [array]  thickness
        -------------------------------------------------------------------------------------
        """

        #Finding the index of the smallest eigenvalue, and finding the respective eigenvector.
        #The smallest eigenvalue is related to the direction of strongest variation of the scalar field, thus to the "thickness" 
        index_min = np.zeros((tot_num))
        emin = np.zeros((2,tot_num))
        for n in range(0,tot_num):
            index_min[n] = np.where(np.abs(eigenvalues2D[:,n]) == np.amin(np.abs(eigenvalues2D[:,n])))[0][0]
            emin[:,n] = eigenvectors2D[:,int(index_min[n]),n]

        #Defining the segment over which we want to interpolate 
        coords_min = np.zeros((2,1000,tot_num))
        ds_min = np.zeros((tot_num))
        for n in range(0,tot_num):
            xin = int(coordx[n])-100*emin[0,n]
            xfin = int(coordx[n])+100*emin[0,n]
            if (xin <= 0) or (xfin <= 0):
                xin = xin + nx
                xfin = xfin + nx
            yin = int(coordy[n])-100*emin[1,n]
            yfin = int(coordy[n])+100*emin[1,n]
            if (yin <= 0) or (yfin <= 0):
                yin = yin + ny
                yfin = yfin + ny
            dst = sp.distance.euclidean([xin,yin], [xfin,yfin])
            ds_min[n] = dst*np.sqrt(dl[0]**2+dl[1]**2)
            coords_min[0,:,n] = np.linspace(xin,xfin,1000)
            coords_min[1,:,n] = np.linspace(yin,yfin,1000)

        #interpolation
        prova_min = np.zeros((1000,tot_num))
        for n in range(0,tot_num):
            prova_min[:,n] = ndm.map_coordinates(AA[:,:],np.vstack((coords_min[0,:,n],coords_min[1,:,n])))

        #--------------
        #Finding the index of the biggest eigenvalue, and finding the respective eigenvector.
        #The biggest eigenvalue is related to the direction perpendicular to the direction of strongest variation, thus to the "width"
        index_max = np.zeros((tot_num))
        emax = np.zeros((2,tot_num))
        for n in range(0,tot_num):
            index_max[n] = np.where(np.abs(eigenvalues2D[:,n]) == np.amax(np.abs(eigenvalues2D[:,n])))[0][0]
            emax[:,n] = eigenvectors2D[:,int(index_max[n]),n]

        coords_max = np.zeros((2,1000,tot_num))
        ds_max = np.zeros((tot_num))
        for n in range(0,tot_num):
            xin = int(coordx[n])-100*emax[0,n]
            xfin = int(coordx[n])+100*emax[0,n]
            if (xin <= 0) or (xfin <= 0):
                xin = xin + nx
                xfin = xfin + nx
            yin = int(coordy[n])-100*emax[1,n]
            yfin = int(coordy[n])+100*emax[1,n]
            if (yin <= 0) or (yfin <= 0):
                yin = yin + ny
                yfin = yfin + ny
            dst = sp.distance.euclidean([xin,yin], [xfin,yfin])
            ds_max[n] = dst*np.sqrt(dl[0]**2+dl[1]**2)
            coords_max[0,:,n] = np.linspace(xin,xfin,1000)
            coords_max[1,:,n] = np.linspace(yin,yfin,1000)

        #interpolation
        prova_max = np.zeros((1000,tot_num))
        for n in range(0,tot_num):
            prova_max[:,n] = ndm.map_coordinates(AA[:,:],np.vstack((coords_max[0,:,n],coords_max[1,:,n])))

        #----------------------
        #CALCULATING FULL WIDTH HALF MAXIMUM FOR THICKNESS
        FWHM_thickness = np.zeros((tot_num))
        for n in range(0,tot_num):
            peaks = scip.find_peaks(prova_max[:,n])
            if (np.shape(peaks[0][:])[0] > 1):        
                fwhm = scip.peak_widths(prova_max[:,n],peaks[0][:])[0]
                altezze = scip.peak_widths(prova_max[:,n],peaks[0][:])[1]
                fwhm = fwhm*ds_max[n]/1000  #conversion in box unit length
                inn = np.where(np.abs(peaks[0][:]-500) == np.amin(np.abs(peaks[0][:]-500)))[0][0]
                FWHM_thickness[n] = fwhm[inn]
            else:
                half = (np.amax(prova_max[400:600,n])-np.amin(prova_max[:,n]))/2 + np.amin(prova_max[:,n])
                max_position = np.where(prova_max[:,n]==np.amax(prova_max[400:600,n]))
                first_half = prova_max[0:max_position[0][0],n]
                second_half = prova_max[max_position[0][0]:-1,n]
                cross_line = np.zeros((1000))
                cross_line[:] = half
                cross_line_1 = np.zeros((max_position[0][0]))
                cross_line_1[:] = half
                cross_line_2 = np.zeros((1000-max_position[0][0]))
                cross_line_2 = half
                s = np.linspace(0,ds_max[n],1000)
                idx_1 = np.where(np.abs((first_half[:] - cross_line_1)) == np.amin(np.abs(first_half[:] - cross_line_1)))
                idx_2 = max_position[0][0]+np.where(np.abs((second_half[:] - cross_line_2)) == np.amin(np.abs(second_half[:] - cross_line_2)))
                FWHM_thickness[n] = np.abs(s[idx_2] - s[idx_1])[0][0]

        #CALCULATING FULL WIDTH HALF MAXIMUM FOR WIDTH
        FWHM_width = np.zeros((tot_num))
        for n in range(0,tot_num):
            peaks = scip.find_peaks(prova_min[:,n])
            if (np.shape(peaks[0][:])[0] > 1):
                fwhm = scip.peak_widths(prova_min[:,n],peaks[0][:])[0]
                altezze = scip.peak_widths(prova_min[:,n],peaks[0][:])[1]
                fwhm = fwhm*ds_min[n]/1000
                inn = np.where(np.abs(peaks[0][:]-500) == np.amin(np.abs(peaks[0][:]-500)))[0][0]
                FWHM_width[n] = fwhm[inn]
            else:
                half = (np.amax(prova_min[400:600,n])-np.amin(prova_min[:,n]))/2 + np.amin(prova_min[:,n])
                max_position = np.where(prova_min[:,n]==np.amax(prova_min[400:600,n]))
                first_half = prova_min[0:max_position[0][0],n]
                second_half = prova_min[max_position[0][0]:-1,n]
                cross_line = np.zeros((1000))
                cross_line[:] = half
                cross_line_1 = np.zeros((max_position[0][0]))
                cross_line_1[:] = half
                cross_line_2 = np.zeros((1000-max_position[0][0]))
                cross_line_2 = half
                s = np.linspace(0,ds_min[n],1000)
                idx_1 = np.where(np.abs((first_half[:] - cross_line_1)) == np.amin(np.abs(first_half[:] - cross_line_1)))
                idx_2 = max_position[0][0]+np.where(np.abs((second_half[:] - cross_line_2)) == np.amin(np.abs(second_half[:] - cross_line_2)))
                FWHM_width[n] = np.abs(s[idx_2] - s[idx_1])[0][0]

        return FWHM_thickness#, FWHM_width

#-------------------------------------------------------------------------------------------
def structure_width(self,
        tot_num, 
        mask_n,
        nx,
        ny,
        nz,
        dl,
        base_value): 
        """
        -------------------------------------------------------------------------------------
        calculates the width of a structure as maximum distance between two points which belong to the same structure
        -------------------------------------------------------------------------------------
        tot_num             [int]  total number of the points over which you want to compute the H matrix
        mask_n              [array] mask of the interesting structures 
        nx                  [int]  box first dimension
        ny                  [int]  box second dimension
        nz                  [int]  box third dimension
        dl                  [float,float,float]  box resolution
        base_value          [int]  value of the ground of the mask (value of all point which don't belong to any structure); 0 or -1
        -------------------------------------------------------------------------------------
        new_var             [array]  width
        -------------------------------------------------------------------------------------
        """

        width = np.zeros((tot_num))
        for ii in range(base_value+1,tot_num+1+base_value):           
            points_x, points_y = np.where(mask_n[:,:]==ii)
            if (np.logical_and(np.any(np.where(points_x==(nx-1))),np.any(np.where(points_x==0)))==True):
                for n in range(0,len(points_x)):
                    if (points_x[n] < nx/2):
                        points_x[n] = points_x[n] + nx
            if (np.logical_and(np.any(np.where(points_y==(ny-1))),np.any(np.where(points_y==0)))==True):
                for n in range(0,len(points_y)):
                    if (points_y[n] < ny/2):
                        points_y[n] = points_y[n] + ny
            lunghezza = len(points_x)
            if (lunghezza > 1):
                if (lunghezza<70000):
                    points = np.zeros((lunghezza,2))
                    points[:,0] = points_x*dl[0]
                    points[:,1] = points_y*dl[1]
                    distance_matrix = sp.distance.pdist(points)
                    width[ii-1] = np.amax(distance_matrix) #NB its in physical unit
                else:
                    #sqrt((xmax-xmin)**2+(ymax-ymin)**2)
                    dist = np.sqrt(((np.amax(points_x)-np.amin(points_x))*dl[0])**2+((np.amax(points_y)-np.amin(points_y))*dl[1])**2)
                    width[ii-1] = dist
            else:  
                width[ii-1] = 0

        return width


