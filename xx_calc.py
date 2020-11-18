"""
The goal of this module is to compute all the quantities required.
--------------------------------------------------------------------------------
Basic fields (B, Ve, n, and E) must be loaded in the format [3,nx,ny] (vectors) 
and [nx,ny] (scalars), where nx and ny are the grid dimensions. Vector fields
are in the form (Vx, Vy, Vz) = [V[0], V[1], V[2]], so that V[2] is
the out-of-plane component.
--------------------------------------------------------------------------------
The final quantities computed by the routine are:
--> the magnitude of the current density                    CALLED Jn
--> |ve_plane|                                              CALLED mod_Ve_plane
--> the magnitude of the z-component of the electric field  CALLED aEdec
--> the magnitude of the curl of the electron velocity      CALLED Omega_ve
--> the magnitude of the in-plane magnetic field            CALLED aB_inplane
--> the energy conversion term                              CALLED JE
--------------------------------------------------------------------------------
The output is a Pandas DataFrame.
--------------------------------------------------------------------------------
m. sisti, f. finelli - xx/xx/xxxx
e-mail: francesco.finelli@phd.unipi.it
"""

#===============================================================================
#IMPORTS
import numpy as np

#===============================================================================
# ... FUNCTIONS


#===============================================================================
#INITIALIZATION
def d_md_init(B,Ve,E,n,run_name='Unnamed Run'):
    """
    INPUT:
        B, Ve, and E -> lists or arrays w/ shape [3,nx,ny]
        n -> list or array w/ shape [nx,ny]
        run_name -> string

    OUTPUT:
        data (dict containig np.array)
        metadata (dict)
    """
#
    data = {'B' : None, 'Ve' : None, 'n' : None, 'E' : None,
            'Jn' : None,
            'mod_Ve_plane' : None,
            'aEdec' : None,
            'Omega_ve' : None,
            'aB_inplane' : None,
            'JE' : None}
    metadata = {'GridSize' : None, 'RunName' : None}
#
    data['B'] = B
    data['Ve'] = Ve
    data['n'] = n
    data['E'] = E
#
    sh_dict = {}
    for key in ['B','Ve','n','E']:
        if type(data[key]) != np.ndarray:
            data[key] = np.array(data[key])
        sh_dict[key] = data[key].shape
#
    for key in ['B','Ve','E']:
        if sh_dict[key][0] != 3:
            print('\nERROR: %s has not the shape [3,nx,ny]!'%key)
            return -1,None
        if sh_dict[key] != sh_dict['n']:
            print('\nERROR: %s does not share the same grid with n!'%key)
            return -1,None
#
    nnn = list(sh_dict['n'])
    nunDims = len(nnn)
    if numDims != 2:
        print('\nERROR: looks like data are %dD!
               \nThis program is intended for 2D.'%numDims)
        return -1,None
#
    metadata['GridSize'] = nnn
    metadata['RunName'] = run_name
#
    return data,metadata

#===============================================================================
#COMPUTE QUANTITIES
# -> in-plane module
def inplane_module(data,key,out_key=None):
    if out_key == None:
        return np.sqrt(np.square(data[key][0])+np.square(data[key][1]))
    else:
        data[out_key] =  np.sqrt(np.square(data[key][0])+np.square(data[key][1]))
        return

# -> cross product (numpy -> dict wrap)
def calc_cross(data,key_A,key_B,out_key=None):
    if out_key == None:
        return np.cross(data[key_A],data[key_B],axis=0)
    else:
        data[out_key] = np.cross(data[key_A],data[key_B],axis=0)
        return

# -> gradient
def calc_grad(data,key,axis,key_out=None,periodic=True):
    oo_6 = dl[1]**(-1) * np.array([1.0,  0.0])
    aa_6 = dl[1]**(-1) * np.array([0.0,  9.0/12.0])
    bb_6 = dl[1]**(-1) * np.array([0.0, -3.0/20.0])
    cc_6 = dl[1]**(-1) * np.array([0.0,  1.0/60.0])

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




# -> mod_Ve_plane
inplane_module(data,'Ve','mod_Ve_plane')

# -> aEdec
data['aEdec'] = np.abs(data['E']+calc_cross(data,'Ve','B'))

# -> Omega_ve
data['Omega_ve'] = rotore di Ve
