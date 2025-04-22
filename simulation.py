import scipy
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from efc_corr_fcns import *
from simulation_fcns import *


file = 'hcp/hcp200.mat'
mat = scipy.io.loadmat(file)
label_num = mat['lab']
system = np.array([e for tupl1 in mat['net'] for tupl2 in tupl1 for e in tupl2])
dictionary = {A: B for A, B in zip(range(1,17), system)}
label = np.vectorize(dictionary.get)(label_num)

idx = []
total = 0
for l in np.unique(label):
    idx_full = np.where(label.ravel() == l)
    np.random.seed(0)
    if len(idx_full[0]) == 14:
        idx_sub = np.random.choice(idx_full[0], size =3, replace=False).ravel()
    else:
        idx_sub = np.random.choice(idx_full[0], size =round(len(idx_full[0])/4), replace=False).ravel()
    idx.append(idx_sub)
    total += len(idx_sub)
idx_subnodes = np.concatenate(idx, axis=0)


p = 50
sum_mean = np.zeros([p])
sum_cov = np.zeros([p, p])

for j in ["%.2d" % i for i in range(1, 11)]:
    # data preprocessing based on read_in_data.m
    for sub in ["%.2d" % i for i in range(1, 11)]:
        file = 'msc/fc/sub-MSC'+str(sub)+'/sub-MSC'+str(sub)+'.ses-func'+j+'_task-rest_out.schaefer200-yeo17.regress_36pNS.mat'
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

        # cleaned data set for p = 50
        ts_low_motion = ts_zs[keep_frames.flatten(),:]
        ts_low_motion = ts_low_motion[:,idx_subnodes]
        n, p = ts_low_motion.shape

        # sample mean, covariance matrix sums
        sum_mean += np.mean(ts_low_motion, axis=0) # check it's 0
        sum_cov += np.dot(ts_low_motion.T, ts_low_motion)/(n-1)


mean = sum_mean / 100
cov = sum_cov / 100
# true_corr_edge = corr_edge(cov, mean)


# make sparse cov
import copy
mean_temp = copy.deepcopy(mean)
cov_temp = copy.deepcopy(cov)
thresholded = np.absolute(cov_temp) < 0.2
cov_temp[thresholded] = 0.0
row, col = np.tril_indices(cov.shape[0], 1) #upper triangular row, col indices
cov_temp[row, col] = 0

cov_temp = cov_temp + cov_temp.transpose()

# eigendecomposition
Lambda, U = np.linalg.eig(cov_temp)
np.fill_diagonal(cov_temp, abs(np.min(Lambda))+0.01)
cov_temp = cov2cor(cov_temp)

true_corr_edge = corr_edge(cov_temp, mean_temp)


# Frobenius norm
def run_simulation(seed, n, C, mean, cov, true_corr_edge):
    fnorms = simulation_efc_corr_reg(seed, mean_temp, cov_temp, true_corr_edge, n, Cs)
    directory = f"/numerical_study/temp_results/fro_sparse_p{p}_n{n}/C{C}"
    os.makedirs(directory, exist_ok=True)
    filename = f"{directory}/iter{seed}.csv"
    pd.DataFrame(np.array([fnorms])).to_csv(filename)


ns = np.arange(50, 550, 50)
Cs = [0.002, 0.005, 0.01]

tasks = ((seed, n, C, mean, cov, true_corr_edge) for n in ns for C in Cs)
with ProcessPoolExecutor() as executor:
    futures = {executor.submit(run_simulation, *task) for task in tasks}
    for future in as_completed(futures):
        future.result()

    
# ROC
rocs = []
ns = [50, 100, 200, 500, 1000]
for i in np.arange(len(ns)):
    roc = threshold_roc(rep, mean_temp,cov_temp, true_corr_edge, ns[i],reverseC=10000)
    rocs.append(roc)



