#---------------------------------------------------
#LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import time

from sklearn import metrics
from sklearn.cluster import KMeans as KM
from sklearn import preprocessing
from sklearn.model_selection import KFold

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

labels_true = None

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
# -> correlation matrix w/ plot
corr = df.corr()

plt.close('all')
fig, ax = plt.subplots(figsize=(15,8))
hm = sns.heatmap(corr,
                 ax=ax,
                 cmap='coolwarm',
                 vmin=-1.0,
                 vmax=1.0,
                 annot=True,
                 fmt='.2f',
                 annot_kws={'size':15},
                 linewidths=0.5)
ax.set_title('Correlation matrix')
plt.savefig(os.path.join(out_dir,'corr_matr_'+ds_name+'.png'))
plt.close()

print('\nCorrelation plot done!')

# -> select data
X = df.loc[:,myvars]
print('\nSelected data:')
print(X.head())
del df

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
# -> ask for k range
kmin = int(input('\nMinimum value for k: '))
kmax = int(input('\nMaximum value for k: '))
krange = range(kmin,kmax+1)

# -> crossvalidation parameters
Nkf = int(input('\nChoose the number of k-folding: '))
n_init = int(input('\nChoose number of initial centroid seeds to try each time: '))
max_iter = 300
n_jobs = min(n_init,n_proc) #parallelization, but only on n_iter
alg_vec = ['full','elkan'] #k-means algorithms

# -> k-fold init.
kf = KFold(n_splits=Nkf,shuffle=True,random_state=rnd_state)
train_sets = []
for train_indices,_ in kf.split(X.loc[:,myvars[0]]):
    train_sets.append(train_indices)

# -> init. output
fname = 'KM_CV_'+ds_name+'_'
fname = fname + 'kmin%dkmax%dnkf%dninit%d'%(kmin,kmax,Nkf,n_init)
print('\nChoose how to open the output file:')
print('w -> creates a new file if it does not exist or truncates the file if it exists;')
print('a -> open for appending at the end of the file without truncating it; creates a new file if it does not exist.')
opopt = input('Insert \'w\' or \'a\': ')
CV_file = open(os.path.join(out_dir,fname+'.txt'),opopt)
CV_file.write('time\talg\tk\tfold\titer\tinertia')

# -> scores selection
# inertia == distortion == WSS (always computed)
scores = {}
# 0 -> labels_true, labels_pred
# 1 -> X, labels == X, y_km
scores['AMI'] = (metrics.adjusted_mutual_info_score,0,'Adjusted Mutual Information') # Mutual Information adjusted for chance
scores['AR'] = (metrics.adjusted_rand_score,0,'Adjusted Rand index') # Rand index adjusted for chance
scores['CH'] = (metrics.calinski_harabaz_score,1,'Calinski-Harabasz') # pseudo F statistic or Variance Ratio Criterion
scores['DB'] = (metrics.davies_bouldin_score,1,'Davies-Bouldin') # the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances
scores['C'] = (metrics.completeness_score,0,'Completeness metric') # A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster
scores['FM'] = (metrics.fowlkes_mallows_score,0,'Fowlkes-Mallows index') # similarity of two clusterings of a set of points
scores['H'] = (metrics.homogeneity_score,0,'Homogeneity metric') # A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class
scores['MI'] = (metrics.mutual_info_score,0,'Mutual Information') # a measure of the similarity between two labels of the same data
scores['NMI'] = (metrics.normalized_mutual_info_score,0,'Normalized Mutual Information') # a normalization of the Mutual Information score to scale the results between 0 and 1
scores['S'] = (metrics.silhouette_score,1,'Silhouette Coefficient') # (b-a)/max(a,b), a='mean intra-cluster distance', b='mean nearest-cluster distance', computed for each sample
scores['VM'] = (metrics.v_measure_score,0,'V-measure') # the harmonic mean between homogeneity and completeness (identical to normalized_mutual_info_score with the 'arithmetic' option)

print('\nThe following scores will be computed:\ninertia')
for s in myscores:
    print(scores[s][2])
    CV_file.write('\t%s'%(s))

CV_file.write('\n')

# -> compute and store scores
for alg in alg_vec:
    print('\n---------------------------------------')
    print('---------------------------------------')
    print('\nalgorithm = %s'%(alg))
    for k in krange:
        print('\n---------------------------------------')
        print('\nk = %d'%(k))
        km = KM(n_clusters=k,init='k-means++',n_init=n_init,max_iter=max_iter,
                random_state=rnd_state,n_jobs=n_jobs,algorithm=alg,verbose=0)
        for n in range(Nkf):
            print('\nn = %d'%(n))
            X_train = X.loc[train_sets[n]]
            print('fitting on train set...')
            t_ = time.time()
            km.fit(X_train)
            t_ = time.time() - t_
            actual_n_iter = km.n_iter_
            inertia = km.inertia_
            CV_file.write('%f\t%s\t%d\t%d\t%d\t%f'%(t_,alg,k,n,actual_n_iter,inertia))
            print('predicting labels on all data...')
            y_km = km.predict(X)
            print('computing scores:')
            for s in myscores:
                print('\t%s'%(scores[s][2]))
                if scores[s][1] == 0:
                    if labels_true == None:
                        print('\tERROR: no true labels for this dataset')
                        tmp = np.nan
                    else:
                        tmp = scores[s][0](labels_true,km.labels)
                elif scores[s][1] == 1:
                    tmp = scores[s][0](X,y_km)
                else:
                    print('\tERROR: unable to compute score inputs')
                    tmp = np.nan
                CV_file.write('\t%f'%(tmp))
            CV_file.write('\n')
            CV_file.flush()

CV_file.close()
print('\ncrossvalidation done!')

#---------------------------------------------------
#READ AND PLOT CROSSVALIDATION DATA
# -> read CV data
CV = pd.read_csv(os.path.join(out_dir,fname+'.txt'),sep='\t',header=0)

print('\nMax number of iteration needed:')
print('%d over %d'%(CV['iter'].max(),max_iter))

# -> mean over k-folds
tmp_col = myscores[:]
tmp_col.append('inertia')
tmp_col.append('time')
tmp_col.append('iter')
tmp = CV.loc[CV['fold']==0,tmp_col].values
for n in range(Nkf-1):
    tmp += CV.loc[CV['fold']==(n+1),tmp_col].values

tmp /= float(Nkf)
CV_mean = pd.DataFrame(tmp)
CV_mean.columns = tmp_col

tmp_col = ['alg','k']
tmp = CV.loc[CV['fold']==0,tmp_col].values
CV_mean[tmp_col] = pd.DataFrame(tmp)
del tmp,tmp_col

CV_mean = CV_mean[['time','alg','k','iter','inertia']+myscores]

# -> compute normalized CV data (already fold_averaged)
min_max_scaler = preprocessing.MinMaxScaler()
CV_norm = CV_mean.loc[:,['time','alg','k','iter']]
tmp_col = ['inertia'] + myscores[:]
tmp = CV_mean[tmp_col].values
tmp_scaled = min_max_scaler.fit_transform(tmp)
CV_norm[tmp_col] = pd.DataFrame(tmp_scaled)
del tmp,tmp_scaled

# -> plot scores
fig,ax = plt.subplots(1,3,figsize=(5*3,5))

for alg in alg_vec:
    ax[0].plot(CV_norm[CV_norm['alg']==alg]['k'],#krange,
            CV_norm.loc[CV_norm['alg']==alg,'time']
            ,marker='o',label=alg)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Time to fit')
    ax[0].set_title('Crossvalidation - %s'%ds_name)
    ax[0].legend()
    ax[0].grid(b=None, axis = 'y')

i = 1
for alg in alg_vec:
    for s in tmp_col:
        ax[i].plot(CV_norm[CV_norm['alg']==alg]['k'],#krange,
                CV_norm.loc[CV_norm['alg']==alg,s]
                ,marker='o',label=s)
    ax[i].set_xlabel('Number of clusters')
    ax[i].set_ylabel('Score')
    ax[i].set_title('Algorithm = %s'%(alg))
    ax[i].legend()
    ax[i].grid(b=None, axis = 'y')
    i += 1

fig.tight_layout()
plt.savefig(os.path.join(out_dir,fname+'.png'))
plt.close()
