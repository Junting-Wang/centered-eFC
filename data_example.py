
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import math
from scipy.linalg import eigh
import igraph
from sklearn.metrics.cluster import contingency_matrix
from matplotlib.collections  import PathCollection
from matplotlib.lines  import Line2D
from igraph import *
from efc_corr_fcns import *
from simulation_fcns import *


# read in all subjects
agg_ts_full = {}
nrow_sub_scan = {} # accumulated # rows over scans of each subject

for s in ["%.2d" % m for m in range(1, 11)]:  
    agg_ts = np.empty((0, 200))
    nrow_sub_scan[s] = {}
    nrow = 0
    
    for j in ["%.2d" % i for i in range(1, 11)]:
        
        # data preprocessing based on read_in_data.m
        file = 'msc/fc/sub-MSC'+str(s)+'/sub-MSC'+str(s)+'.ses-func'+str(j)+'_task-rest_out.schaefer200-yeo17.regress_36pNS.mat'
        mat = scipy.io.loadmat(file)
        # get time series data
        ts = mat['parcel_time'][0][0]
        # zscore
        ts_zs = stats.zscore(ts, axis=0, ddof=1)
        num_frames, num_nodes = ts_zs.shape

        # deal with motion
        thr = 0.4  # drop frames with motion > thr
        dilate = 2 # drop if a low-motion frame is within 'dilate' frames of a high-motion frame
        minlen = 5 # keep low-motion frames that form contiguous sequences of at at least 'minlen'
    
        # make a mask of usable frames
        tmask = mat['tmask_all'][0][0]
        keep_frames0 = tmask < thr
        keep_frames = tmask < thr

        for i in range(num_frames):
            idx = range(i - dilate, i + dilate+1)
            idx = [x for x in idx if (x >= 0 and x < num_frames)]

            if any(keep_frames0[idx] == False):
                keep_frames[i] = False
    
        idx = np.where(keep_frames)[0]
        diff = idx - range(len(idx))
        unq = np.unique(diff)
    
        for k in range(len(unq)):
            kdx = diff == unq[k]
            if sum(kdx) < minlen:
                keep_frames[idx[kdx]] = False
    
        # cleaned data set
        ts_low_motion = ts_zs[keep_frames.flatten(),:]
        ts_low_motion_zs = stats.zscore(ts_low_motion, axis=0, ddof=1)
        nrow += ts_low_motion.shape[0]
        nrow_sub_scan[s][j] = nrow
        
        # aggregate fMRI data across all scans
        agg_ts = np.vstack([agg_ts, ts_low_motion_zs])
        
    agg_ts_full[s] = agg_ts



# eFC & eFC-corr from aggregated data (subject 01-10)
agg_efc_corr = {}
for s in ["%.2d" % m for m in range(1, 11)]:  
    agg_edge_ts = fcn_ets(agg_ts_full[s])
    agg_efc_corr[s] = fcn_eFC_corr(agg_edge_ts)


file = 'hcp/hcp200.mat'
mat = scipy.io.loadmat(file)
label_num = mat['lab']
system = np.array([e for tupl1 in mat['net'] for tupl2 in tupl1 for e in tupl2])
dictionary = {A: B for A, B in zip(range(1,17), system)}
label = np.vectorize(dictionary.get)(label_num)


# compute matthews correlation matrix
from sklearn.metrics import matthews_corrcoef
def matthews_corrmat(df1, df2):
    corr_mat = np.zeros((df1.shape[1], df2.shape[1]))
    for i in range(df1.shape[1]):
        for j in range(df2.shape[1]):
            corr_mat[i,j] = matthews_corrcoef(df1.iloc[:, i], df2.iloc[:, j])
    return corr_mat


# Kmeans for average eFC-corr sample estimations over 10 subjects
from sklearn.cluster import KMeans

avg_sample_efc_corr = np.zeros((num_edges, num_edges))
for sub in ["%.2d" % m for m in range(1, 11)]: 
    avg_sample_efc_corr += agg_efc_corr[sub]
    
avg_sample_efc_corr = avg_sample_efc_corr/10.0


p = avg_sample_efc_corr.shape[1]
w, v = eigh(avg_sample_efc_corr, subset_by_index=[p-50, p-1])
eivec_scale = v*1./np.max(abs(v), axis=0)
df_eivec_scale = pd.DataFrame(eivec_scale)

kmeans5 = KMeans(n_clusters=5)
kmeans5.fit(df_eivec_scale)
cluster5 = kmeans5.predict(df_eivec_scale)


# example of projection matrix
# supposed to relabel cluster based on avg regularized estimator
p=200
mat1 = np.zeros((p,p))
mat1[np.triu_indices(mat1.shape[0], k = 1)] = cluster5
mat1 = mat1 + np.transpose(mat1)
np.fill_diagonal(mat1, 100)
proj1 = np.zeros((p,np.max(cluster5)+1))
for i in np.arange(np.max(cluster5)+1):
    mask1 = mat1 == i
    proj1[:,i] = np.sum(mask1,1)/(p-1)
