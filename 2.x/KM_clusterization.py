'''
m. sisti, f. finelli - 18/11/2020
e-mail: manuela.sisti@univ-amu.fr, francesco.finelli@phd.unipi.it

Clusterization using KMeans.
'''

#---------------------------------------------------
#LIBRARIES
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os
import time
import sys

from sklearn.cluster import KMeans as KM
from sklearn import preprocessing

#---------------------------------------------------
#FUNCTIONS
def rs():
    t = int( time.time() * 1000.0 )
    s = ( ((t & 0xff000000) >> 24) +
          ((t & 0x00ff0000) >>  8) +
          ((t & 0x0000ff00) <<  8) +
          ((t & 0x000000ff) << 24)   )
    return s

#---------------------------------------------------
#INITIALIZATION (MOSTLY HARDCODED)
rnd_state = rs()#666#

df_path = '/home/finelli/Downloads/KMeans'
df_name = 'alldata_t*.txt'

vars_ = ["J","x","y","Dummy","V","EV","EVB","CM","Jep"]
myvars = ["J","V","EV","EVB","CM","Jep"]

out_dir = '/home/finelli/Downloads/KMeans'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

n_proc = 5#12#

myscores = ['DB']

preprocessing_opt = 'norm'#'std'#None

#---------------------------------------------------
#READ DATA
# -> choose the file
df_files = glob.glob(os.path.join(df_path,df_name))
if len(df_files) == 0:
    sys.exit("ERROR: no file %s found in:\n"%(df_name)+pvi_path)

print('\nfiles:')
for n,f in zip(range(len(df_files)),df_files):
    print('%d -> %s'%(n,os.path.split(f)[1]))

n = int(input('Choose a file (insert the corresponding number): '))
df_file = df_files[n]
ds_name = os.path.split(df_file)[1].split('.')[0]

# -> read
df = pd.read_csv(df_file,sep="\t",header=None)
df.columns = vars_
print('\nAll data:')
print(df.head())

#---------------------------------------------------
#PREPROCESSING
# -> select data
X = df.loc[:,myvars]
print('\nSelected data:')
print(X.head())

for v in ['Dummy','dummy','Mask','mask']:
    if v in df.columns: df = df.drop(columns=[v])

# -> normalize, standardize, or nothing
if preprocessing_opt == 'norm':
    # ->-> normalize data
    min_max_scaler = preprocessing.MinMaxScaler()
    tmp = X.values
    tmp_scaled = min_max_scaler.fit_transform(tmp)
    X = pd.DataFrame(tmp_scaled)
    X.columns = myvars
    del tmp,tmp_scaled
    print('\nNormalized data:')
    print(X.head())
elif preprocessing_opt == 'std':
    # ->-> stndardize data
    std_scaler = preprocessing.StandardScaler()
    tmp = X.values
    tmp_scaled = std_scaler.fit_transform(tmp)
    X = pd.DataFrame(tmp_scaled)
    X.columns = myvars
    del tmp,tmp_scaled
    print('\nStandardized data:')
    print(X.head())
elif preprocessing_opt == None:
    print('\nData are note scaled/standardized/etc...')
else:
    print('\nERROR: unkown value for preprocessing_opt. Exiting...')
    sys.exit()

#---------------------------------------------------
#PARAMETERS OPTIMIZATION
# -> ask for k
k = int(input('\nValue for k: '))

# -> crossvalidation parameters
n_init = int(input('\nChoose number of initial centroid seeds to try: '))
max_iter = 300
n_jobs = min(n_init,n_proc) #parallelization, but only on n_iter
alg = str(input('\nChoose algorithm (full,elkam): '))#,'elkan'] #k-means algorithms

# -> compute and store scores
print('\n---------------------------------------')
print('---------------------------------------')
print('\nalgorithm = %s'%(alg))
print('\n---------------------------------------')
print('\nk = %d'%(k))
km = KM(n_clusters=k,init='k-means++',n_init=n_init,max_iter=max_iter,
        random_state=rnd_state,n_jobs=n_jobs,algorithm=alg,verbose=0)
print('fitting on full data set...')
km.fit(X)
actual_n_iter = km.n_iter_
inertia = km.inertia_
df['label'] = km.labels_
print('\nclusterization done!')
del X

print('\nclusterization inertia (WSS): '+str(inertia))
print('\nNumber of iteration needed:')
print('%d over %d'%(actual_n_iter,max_iter))
if actual_n_iter == max_iter:
    print('\nWARNING: maximum number of iterations reached before convergence; consider increasing max_iter and retry.')

# -> output
fname = 'KM_clust_'+ds_name+'_'
fname = fname + 'k%dalg%sninit%d'%(k,alg,n_init)
print('\nChoose how to open the output file:')
print('w -> creates a new file if it does not exist or truncates the file if it exists;')
print('a -> open for appending at the end of the file without truncating it; creates a new file if it does not exist.')
opopt = input('Insert \'w\' or \'a\': ')
if opopt == 'w':
    header = 1
elif opopt == 'a':
    header = None

df.to_csv(os.path.join(out_dir,fname+'.txt'),header=header,index=None,sep='\t',mode=opopt)
del df

#---------------------------------------------------
#READ AND PLOT CLUSTERS
# -> read cluster data
clust = pd.read_csv(os.path.join(out_dir,fname+'.txt'),sep='\t',header=0)

clust_x_min = clust['x'].min()
clust_x_max = clust['x'].max()
clust_y_min = clust['y'].min()
clust_y_max = clust['y'].max()
nnx = clust_x_max - clust_x_min + 1
nny = clust_y_max - clust_y_min + 1

clust2D = clust['label'].values.reshape(nnx,nny)

pcmX = np.linspace(clust_x_min-0.5,clust_x_max+0.5,nnx+1)
pcmY = np.linspace(clust_y_min-0.5,clust_y_max+0.5,nny+1)

# -> find reconnections cluster
labels = np.unique(clust['label'].values)
labels.sort()
lmin = labels[0]
lmax = labels[-1]
nlab = len(labels)

Jmeanmax = -1.
lrec = lmin - 1
for lcls in labels:
    #J_ = clust[clust['label']==lcls]['J'].max()*clust[clust['label']==lcls]['J'].mean()
    J_ = (clust[clust['label']==lcls]['J'].mean())
    if J_ > Jmeanmax:
        Jmeanmax = J_
        lrec = lcls

if lrec < lmin:
    print('\nERROR: something went wrong while looking for the \"reconnection\" cluster. Exiting...')
    sys.exit()

# -> plot clust
plt.close('all')
fig,ax = plt.subplots(1,1,figsize=(12,12))

cmap = mpl.cm.get_cmap('viridis',nlab)
newcolors = cmap(np.linspace(0,1,nlab))
pink = np.array([248./256.,24./256.,148./256.,1.])
lrec_idx = np.argmin(np.abs(labels-lrec))
newcolors[lrec_idx:lrec_idx+1,:] = pink
cmap = mpl.colors.ListedColormap(newcolors)

bnd       = np.empty(nlab+1,dtype=np.float32)
bnd[1:-1] = (labels[1:] + labels[:-1])/2.
bnd[0]    = 2*lmin - bnd[1]
bnd[-1]   = 2*lmax - bnd[-2]
norm = mpl.colors.BoundaryNorm(bnd, cmap.N)

im = ax.pcolormesh(pcmX,pcmY,clust2D.T,cmap=cmap,norm=norm)
cbar = plt.colorbar(im,ax=ax,ticks=(bnd[1:]+bnd[:-1])/2.,label='labels')
cbar.ax.set_yticklabels(labels.astype(np.str))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Clusters in spatial domain (pink is the \"probable reconnection\" one)')
plt.tight_layout()
plt.show()
plt.close()
