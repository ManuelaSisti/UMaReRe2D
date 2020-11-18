'''
m. sisti, f. finelli - 18/11/2020
e-mail: manuela.sisti@univ-amu.fr, francesco.finelli@phd.unipi.it

Computing clusters' aspect ratio.
'''

import utilities_unsup as ut

#------------------------------------------------------
#loading maxima and test_mask
cluster_maxima_and_loc = np.zeros((real_cluster_number,2))

file2 = open(file_maxima, 'r')
for i in range(0,real_cluster_number):
    tr = file2.readline().split()
    cluster_maxima_and_loc[i,0] = float(tr[0])
    cluster_maxima_and_loc[i,1] = float(tr[1])

file2.close()

test_mask = np.zeros((nx,ny,2), dtype=int)
test_mask[:,:,0] = np.loadtxt(file_mask_binary)
test_mask[:,:,1] = np.loadtxt(file_mask_clusters)

#------------------------------------------------------
#Computing width and thickness of each cluster

print('Computing width (max distance method)')
width = ut.structure_width(ut,real_cluster_number,test_mask[:,:,1],nx,ny,nz,dl,base_value=0)
H, eigenvalues2D, eigenvectors2D = ut.Hmatrix(ut,Jn[:,:,:],real_cluster_number,cluster_maxima_and_loc[:,0],cluster_maxima_and_loc[:,1],nx,ny,nz,dl,normalization = True)
Jn_per = np.tile(Jn[:,:,0],(2,2))
print('Computing thickness (FWHM method)')
FWHM_thickness = ut.interpolation_for_width_and_thickness(ut,eigenvalues2D,eigenvectors2D,real_cluster_number,cluster_maxima_and_loc[:,0],cluster_maxima_and_loc[:,1],dl,Jn_per,nx,ny,nz)

#Computing aspect ratio
print('Computing aspect ratio (width/thickness)')
aspect_ratio = width/FWHM_thickness

counter = 0
for i in range(0,len(aspect_ratio)): #correction necessary when the cluster is almost circular and width~thickness
    if (aspect_ratio[i] < 1.0):
        aspect_ratio[i] = 1.0/aspect_ratio[i]
        counter = counter + 1

