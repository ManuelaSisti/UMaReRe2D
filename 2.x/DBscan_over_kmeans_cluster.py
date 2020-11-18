'''
m. sisti, f. finelli - 18/11/2020
e-mail: manuela.sisti@univ-amu.fr, francesco.finelli@phd.unipi.it

Applying DBscan to one selected cluster, among those found using KMeans algorithm.
"label_rec" refers to the label of the chosen cluster among those found by KMeans. 
'''

import h2o 
from h2o.estimators import H2OIsolationForestEstimator
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

h2o.init()

#reading file using pandas
h2o_df = h2o.import_file(file_kmeans)
if 'label' in h2o_df.columns:
  h2o_df_work = h2o_df[h2o_df['label']==label_rec]
  h2o_df_work = h2o_df_work.drop(['label'])
else:
  h2o_df_work = h2o_df.loc[:]

coord_sel = h2o_df_work.drop([0,3,4,5,6,7,8], axis=1)

#applying dbscan 
coord_array = coord_sel.as_data_frame().as_matrix()
clustering = DBSCAN(eps=50, min_samples=100).fit(coord_array)
clustering.labels_

#converting again to h2o data frame
to_pandas = pd.DataFrame(clustering.labels_)
clusterized = h2o.H2OFrame(to_pandas, column_names=['clusterization'])
h2o_df_work = h2o_df_work.cbind(clusterized)

#pandas
h2o_to_pandas = h2o_df_work.as_data_frame()

#saving
np.savetxt(file_DBscan,h2o_to_pandas.values, header = 'J\tx\ty\tV\tEV\tEVB\tCM\tJep\tclusterization',fmt='%f %d %d %f %f %f %f %f %d')

#==========================================================================================
#in this second part we apply periodicity to the obtained clusters, if the box is periodic

#reading clusters obtained applying DBscan 
test_mask = np.zeros((nx,ny,2), dtype=int)
file1 = open(file_DBscan, 'r')
file_DBscan_length = len(file1.readlines())
file1.seek(0)
file1.readline()
file1.readline().split()
for i in range(0,file_DBscan_length-2):
    tr = file1.readline().split()
    a = int(tr[1])
    b = int(tr[2])
    test_mask[a,b,0] = 1
    test_mask[a,b,1] = int(tr[8])+1 #+1 is necessary in order to distinguish clusters from the base value (-1 indicates isolated points)

file1.close()

#Merging clusters using periodicity
for ix in range(0,len(test_mask[:,0,1])):
    if (test_mask[ix,0,1] != 0):
        if (test_mask[ix,-1,1] != 0):
             xx, yy = np.where(test_mask[:,:,1]==test_mask[ix,-1,1]) 
             test_mask[xx,yy,1] = test_mask[ix,0,1]

#Merging current sheets using periodicity
for iy in range(0,len(test_mask[0,:,1])):
    if (test_mask[0,iy,1] != 0):
        if (test_mask[-1,iy,1] != 0):
             xx, yy = np.where(test_mask[:,:,1]==test_mask[-1,iy,1]) 
             test_mask[xx,yy,1] = test_mask[0,iy,1]

mmm = np.amax(test_mask[:,:,1])
#print(mmm)

real_cluster_number = 0
for i in range(1,mmm+1):
    xxx,yyy = np.where(test_mask[:,:,1]==i)
    if (np.shape(xxx)[0] !=0):
        real_cluster_number = real_cluster_number + 1
        test_mask[xxx,yyy,1] = real_cluster_number

#-------------------------------------------------------------------------------------
#figure test
plt.figure(figsize=(10,8.5))
plt.contourf(Jn[:,:,0].T,30)
plt.colorbar()
for i in range(1,np.amax(test_mask[:,:,1])+1):
    xindex, yindex = np.where(test_mask[:,:,1]==i)
    plt.plot(xindex,yindex,'.',markersize=1)

plt.title('Clusters (K-means + DBscan), t=%.f, over $|J|$' %time)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show(0)

#-------------------------------------------------------------------------------------
#saving positions in grid points of local maxima for each cluster
#saving test_mask

maxima_cluster = np.zeros((0))
maxima_cluster_x = np.zeros((0))
maxima_cluster_y = np.zeros((0))

for i in range(1,np.amax(test_mask[:,:,1])+1):
    xindex, yindex = np.where(test_mask[:,:,1]==i)
    Jn_cluster = Jn[xindex,yindex,0]
    maximum_cluster = np.amax(Jn_cluster) #WARNING: it is necessary to have uploaded data for the right time
    maximum_cluster_location = np.where(Jn_cluster==maximum_cluster)
    maxima_cluster_x = np.append(maxima_cluster_x,xindex[maximum_cluster_location])
    maxima_cluster_y = np.append(maxima_cluster_y,yindex[maximum_cluster_location])
    maxima_cluster = np.append(maxima_cluster,maximum_cluster)

#writing maxima
file1 = open(file_maxima, 'w')
for i in range(0,len(maxima_cluster)):
    tw1 = '%.f' % maxima_cluster_x[i]
    tw2 = '%.f' % maxima_cluster_y[i]
    file1.write(tw1 + '\t' + tw2 + '\n')

file1.close()

#writing test_mask
np.savetxt(file_mask_binary,test_mask[:,:,0])
np.savetxt(file_mask_clusters,test_mask[:,:,1])

