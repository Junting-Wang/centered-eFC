import scipy
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import copy
from efc_corr_fcns import *
from simulation_fcns import *
from test_stat_fcns import *

p = 50
mean = np.zeros(p)
cov = np.zeros((p,p))
row, col = np.tril_indices(cov.shape[0], -1) #lower triangular row, col indices
np.random.seed(p*2)
nonzero_idx = np.random.choice(np.arange(len(row)), size=math.floor(0.05*(p**2-p)/2), replace=False)
# nonzero_row = np.random.choice(row, size=math.floor(0.8*(p**2-p)/2), replace=False)
# nonzero_col = np.random.choice(col, size=math.floor(0.8*(p**2-p)/2), replace=False)
cov[row[nonzero_idx], col[nonzero_idx]] =0.9
cov
cov = cov + cov.transpose()
# eigendecomposition
Lambda, U = np.linalg.eig(cov)
print(np.min(Lambda))
np.fill_diagonal(cov, abs(np.min(Lambda))+0.1)
cov = cov2cor(cov)

true_corr_edge = corr_edge(cov, mean)



# Type I & Power
# test stat for 0 ~ 0.2 & 0.2 ~ 0.5
alpha = 0.05

np.random.seed(seed1)
X = np.random.multivariate_normal(mean, cov, n)
ets = fcn_ets(X)
eFC_corr = fcn_eFC_corr(ets)

# test stat for 0
check_gamma_zeros = test_stat_subset_zeros(X, ets, eFC_corr, true_corr_edge, num_entries=1000, seed=seed2)
type1err_pval = stats.norm.sf(abs(check_gamma_zeros))*2
type1errs = ((type1err_pval < alpha).sum())/1000

# test stat for 0 ~ 0.2
check_gamma_nonzeros0_02 = test_stat_subset_nonzeros_range(X, ets, eFC_corr, true_corr_edge, num_entries=1000, range1=0.0, range2=0.2, seed=seed2)
power_pval0_02 = stats.norm.sf(abs(check_gamma_nonzeros0_02))*2
powers_0_02 = ((power_pval0_02 < alpha).sum())/1000

#         type1err0_02 = type1err_subset(check_gamma_zeros0_02,eFC_corr, true_corr_edge, alpha)
#         power0_02 = power_subset(check_gamma_nonzeros0_02, eFC_corr, true_corr_edge, alpha)

# test stat for 0.2 ~ 0.5
check_gamma_nonzeros02_05 = test_stat_subset_nonzeros_range(X, ets, eFC_corr, true_corr_edge, num_entries=1000, range1=0.2, range2=0.5, seed=seed2)
power_pval02_05 = stats.norm.sf(abs(check_gamma_nonzeros02_05))*2
powers_02_05 = ((power_pval02_05 < alpha).sum())/1000

#         type1err02_05 = type1err_subset(check_gamma_zeros02_05,eFC_corr, true_corr_edge, alpha)
#         power02_05 = power_subset(check_gamma_nonzeros02_05, eFC_corr, true_corr_edge, alpha)

# FDR
check_gamma_zeros, check_gamma_nonzeros = fdr_test_stat_subset_std_ets_sparsity(X, ets, eFC_corr, true_corr_edge, num_entries=3000, range1=0.0, seed=seed2)
fdr = fdr_subset(check_gamma_zeros, check_gamma_nonzeros, eFC_corr, true_corr_edge, alpha)
  
