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


def fdr_subset(gamma_zeros, gamma_nonzeros, eFC_corr, true_corr_edge, alpha):
    '''
    input:
        gamma: test statistics under FDR procedure (num of selected entries from upper triangular in eFC_corr)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        true_corr_edge: true edge correlation matrix
        alpha: prespecified (edge)x(edge)
    return:
        FDR based on selected entries
    '''

    gamma = np.concatenate((gamma_zeros, gamma_nonzeros))
    N = len(gamma)
    L = np.sum(1/(np.arange(N) + 1))
    pval = stats.norm.sf(abs(gamma))*2
    pval_sort = np.sort(pval)
    idx = np.argsort(pval)
    num_edges = np.shape(true_corr_edge)[1]
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1))
    
    for l in (np.arange(N)+1):
        if pval_sort[l-1] > l*alpha/(N*L):
            r = l # num of rejections
            break
    
    rej_idx = idx[:r] # idx for rejections in pval
    # first nums of idx_random_zeros corresponds to true H0
    idx_FP = [i for i in rej_idx if i < len(gamma_zeros)]
    num_FP = len(idx_FP)
    FDR = num_FP/r
    return FDR


def test_stat_subset_zeros(X, ets, eFC_corr, true_corr_edge, num_entries, seed):
    '''
    input:
        X: data (n sample)x(p)
        ets: edge time series (time)x(edge)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        num_entries: number of selected entries from upper triangular of eFC_corr
    return: 
        gamma: test statistics based on zero entries for type 1 error (num_entries)
    '''
    n = np.shape(X)[0] # num of samples
    p = np.shape(X)[1] # num of nodes
    num_edges = np.shape(eFC_corr)[1] # num of edges
    x_zs = stats.zscore(X, axis=0, ddof = 1)
    
    # upper tringular idx for edges
    idx_row_ets, idx_col_ets = np.nonzero(np.triu(np.ones((p, p)), 1))
    # upper tringular idx from eFC_corr
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1)) 
    
    # col i in ets_center = X col [idx_row_ets[i]] times X col [idx_col_ets[i]]
    ets = stats.zscore(ets, axis=0, ddof = 1)
    
    # get index of upper triangular entries who are ZERO
    idx_zeros = np.where(true_corr_edge[idx_row, idx_col] == 0)
    
    # randomly select subsets from zero entries from true edge corr
    ########### change size to num of entries
    np.random.seed(seed)
    idx_random_zeros = np.random.choice(np.asarray(idx_zeros).ravel(), size=int(num_entries), replace=False)
#     idx_random_nonzeros = np.random.choice(np.asarray(idx_nonzeros).ravel(), size=int(num_entries), replace=False)
    idx_row_zeros, idx_col_zeros = idx_row[idx_random_zeros], idx_col[idx_random_zeros]
#     idx_row_nonzeros, idx_col_nonzeros = idx_row[idx_random_nonzeros], idx_col[idx_random_nonzeros]
        
    gamma_zeros =  np.zeros(len(idx_random_zeros))
#     gamma_nonzeros = np.zeros(len(idx_random_nonzeros))

    iteration1 = -1
    for i in idx_random_zeros:
        iteration1 += 1
        # compute Lambda_jkst from selected zero entry from true edge corr
        idx_jk, idx_st = idx_row[i], idx_col[i]
        Var_jkst = np.var(ets[:,idx_jk]*ets[:,idx_st], ddof=1)

        sigma_jk = np.mean(ets[:,idx_jk])
        sigma_st = np.mean(ets[:,idx_st])

        Var_jk = np.var(ets[:,idx_jk], ddof=1)
        Var_st = np.var(ets[:,idx_st], ddof=1)

        X_jk = ets[:,idx_jk]
        X_st = ets[:,idx_st]
        X_jkst = ets[:,idx_jk] * ets[:,idx_st]
        X_jk_jkst = X_jk * X_jkst
        X_st_jkst = X_st * X_jkst
        Cov_jk_jkst = np.mean(X_jk_jkst) - np.mean(X_jk) * np.mean(X_jkst)
        Cov_st_jkst = np.mean(X_st_jkst) - np.mean(X_st) * np.mean(X_jkst)
        Cov_jk_st = np.mean(X_jkst) - np.mean(X_jk) * np.mean(X_st)

        lambdahat_jkst = Var_jkst + (sigma_st**2)*Var_jk + (sigma_jk**2)*Var_st - 2*sigma_st*Cov_jk_jkst \
                            - 2*sigma_jk*Cov_st_jkst + 2*sigma_jk*sigma_st*Cov_jk_st
#         lambdahat_jkst = Var_jkst
        gamma_jkst = eFC_corr[idx_row[i], idx_col[i]]/math.sqrt(lambdahat_jkst/n)
        gamma_zeros[iteration1] = gamma_jkst
        
    return gamma_zeros



def test_stat_subset_nonzeros_range(X, ets, eFC_corr, true_corr_edge, num_entries, range1, range2, seed):
    '''
    input:
        X: data (n sample)x(p)
        ets: edge time series (time)x(edge)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        num_entries: number of selected entries from upper triangular of eFC_corr
    return: 
        gamma: test statistics based on nonzeros within range (num_entries)
    '''
    n = np.shape(X)[0] # num of samples
    p = np.shape(X)[1] # num of nodes
    num_edges = np.shape(eFC_corr)[1] # num of edges
    x_zs = stats.zscore(X, axis=0, ddof = 1)
    
    # upper tringular idx for edges
    idx_row_ets, idx_col_ets = np.nonzero(np.triu(np.ones((p, p)), 1))
    # upper tringular idx from eFC_corr
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1)) 
    
    # col i in ets_center = X col [idx_row_ets[i]] times X col [idx_col_ets[i]]
    ets = stats.zscore(ets, axis=0, ddof = 1)
    
    # get index of upper triangular entries who are zero or within the RANGE
#     idx_zeros = np.where(true_corr_edge[idx_row, idx_col] == 0)
    idx_nonzeros = np.where((true_corr_edge[idx_row, idx_col]>range1) & (true_corr_edge[idx_row, idx_col]<range2))

    # randomly select subsets retaining sparsity from zero & nonzero entries from true edge corr
    ########### change size to num of entries (without sparsity)
    np.random.seed(seed)
#     idx_random_zeros = np.random.choice(np.asarray(idx_zeros).ravel(), size=int(num_entries), replace=False)
    idx_random_nonzeros = np.random.choice(np.asarray(idx_nonzeros).ravel(), size=int(num_entries), replace=False)
#     idx_row_zeros, idx_col_zeros = idx_row[idx_random_zeros], idx_col[idx_random_zeros]
    idx_row_nonzeros, idx_col_nonzeros = idx_row[idx_random_nonzeros], idx_col[idx_random_nonzeros]
        
#     gamma_zeros =  np.zeros(len(idx_random_zeros))
    gamma_nonzeros = np.zeros(len(idx_random_nonzeros))
    
    iteration2 = -1
    for i in idx_random_nonzeros:
        iteration2 += 1
        # compute Lambda_jkst from selected zero entry from true edge corr
        idx_jk, idx_st = idx_row[i], idx_col[i]
        Var_jkst = np.var(ets[:,idx_jk]*ets[:,idx_st], ddof=1)

        sigma_jk = np.mean(ets[:,idx_jk])
        sigma_st = np.mean(ets[:,idx_st])

        Var_jk = np.var(ets[:,idx_jk], ddof=1)
        Var_st = np.var(ets[:,idx_st], ddof=1)

        X_jk = ets[:,idx_jk]
        X_st = ets[:,idx_st]
        X_jkst = ets[:,idx_jk] * ets[:,idx_st]
        X_jk_jkst = X_jk * X_jkst
        X_st_jkst = X_st * X_jkst
        Cov_jk_jkst = np.mean(X_jk_jkst) - np.mean(X_jk) * np.mean(X_jkst)
        Cov_st_jkst = np.mean(X_st_jkst) - np.mean(X_st) * np.mean(X_jkst)
        Cov_jk_st = np.mean(X_jkst) - np.mean(X_jk) * np.mean(X_st)

        lambdahat_jkst = Var_jkst + (sigma_st**2)*Var_jk + (sigma_jk**2)*Var_st - 2*sigma_st*Cov_jk_jkst \
                            - 2*sigma_jk*Cov_st_jkst + 2*sigma_jk*sigma_st*Cov_jk_st
#         lambdahat_jkst = Var_jkst
        gamma_jkst = eFC_corr[idx_row[i], idx_col[i]]/math.sqrt(lambdahat_jkst/n)
        gamma_nonzeros[iteration2] = gamma_jkst
    
    return gamma_nonzeros



def fdr_test_stat_subset_std_ets_sparsity(X, ets, eFC_corr, true_corr_edge, num_entries, range1, seed):
    '''
    input:
        X: data (n sample)x(p)
        ets: edge time series (time)x(edge)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        num_entries: number of selected entries from upper triangular of eFC_corr (retaining sparsity)
    return: 
        gamma: test statistics under FDR procedure (num of upper triangular entries in eFC_corr)
    '''
    n = np.shape(X)[0] # num of samples
    p = np.shape(X)[1] # num of nodes
    num_edges = np.shape(eFC_corr)[1] # num of edges
    x_zs = stats.zscore(X, axis=0, ddof = 1)
    
    # upper tringular idx for edges
    idx_row_ets, idx_col_ets = np.nonzero(np.triu(np.ones((p, p)), 1))
    # upper tringular idx from eFC_corr
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1)) 
    
    # col i in ets_center = X col [idx_row_ets[i]] times X col [idx_col_ets[i]]
    ets = stats.zscore(ets, axis=0, ddof = 1)
    
    # get index of upper triangular entries who are zero or not
    idx_zeros = np.where(true_corr_edge[idx_row, idx_col] == 0)
    idx_nonzeros = np.where(true_corr_edge[idx_row, idx_col] > range1)
#     sparsity = len(idx_zeros[0])/len(idx_row)

    
    # randomly select subsets retaining sparsity from zero & nonzero entries from true edge corr
    # select WITH sparsity
    np.random.seed(seed)
    idx_random_zeros = np.random.choice(np.asarray(idx_zeros).ravel(), size=int(num_entries*int(len(idx_zeros[0])/len(idx_row)*100)/100), replace=False)
    idx_random_nonzeros = np.random.choice(np.asarray(idx_nonzeros).ravel(), size=int(num_entries - num_entries*int(len(idx_zeros[0])/len(idx_row)*100)/100), replace=False)
    idx_row_zeros, idx_col_zeros = idx_row[idx_random_zeros], idx_col[idx_random_zeros]
    idx_row_nonzeros, idx_col_nonzeros = idx_row[idx_random_nonzeros], idx_col[idx_random_nonzeros]
    
    gamma_zeros =  np.zeros(len(idx_random_zeros))
    gamma_nonzeros = np.zeros(len(idx_random_nonzeros))

    iteration1 = -1
    for i in idx_random_zeros:
        iteration1 += 1
        # compute Lambda_jkst from selected zero entry from true edge corr
        idx_jk, idx_st = idx_row[i], idx_col[i]
        Var_jkst = np.var(ets[:,idx_jk]*ets[:,idx_st], ddof=1)

        sigma_jk = np.mean(ets[:,idx_jk])
        sigma_st = np.mean(ets[:,idx_st])

        Var_jk = np.var(ets[:,idx_jk], ddof=1)
        Var_st = np.var(ets[:,idx_st], ddof=1)

        X_jk = ets[:,idx_jk]
        X_st = ets[:,idx_st]
        X_jkst = ets[:,idx_jk] * ets[:,idx_st]
        X_jk_jkst = X_jk * X_jkst
        X_st_jkst = X_st * X_jkst
        Cov_jk_jkst = np.mean(X_jk_jkst) - np.mean(X_jk) * np.mean(X_jkst)
        Cov_st_jkst = np.mean(X_st_jkst) - np.mean(X_st) * np.mean(X_jkst)
        Cov_jk_st = np.mean(X_jkst) - np.mean(X_jk) * np.mean(X_st)

        lambdahat_jkst = Var_jkst + (sigma_st**2)*Var_jk + (sigma_jk**2)*Var_st - 2*sigma_st*Cov_jk_jkst \
                            - 2*sigma_jk*Cov_st_jkst + 2*sigma_jk*sigma_st*Cov_jk_st
#         lambdahat_jkst = Var_jkst
        gamma_jkst = eFC_corr[idx_row[i], idx_col[i]]/math.sqrt(lambdahat_jkst/n)
        gamma_zeros[iteration1] = gamma_jkst
    
    iteration2 = -1
    for i in idx_random_nonzeros:
        iteration2 += 1
        # compute Lambda_jkst from selected zero entry from true edge corr
        idx_jk, idx_st = idx_row[i], idx_col[i]
        Var_jkst = np.var(ets[:,idx_jk]*ets[:,idx_st], ddof=1)

        sigma_jk = np.mean(ets[:,idx_jk])
        sigma_st = np.mean(ets[:,idx_st])

        Var_jk = np.var(ets[:,idx_jk], ddof=1)
        Var_st = np.var(ets[:,idx_st], ddof=1)

        X_jk = ets[:,idx_jk]
        X_st = ets[:,idx_st]
        X_jkst = ets[:,idx_jk] * ets[:,idx_st]
        X_jk_jkst = X_jk * X_jkst
        X_st_jkst = X_st * X_jkst
        Cov_jk_jkst = np.mean(X_jk_jkst) - np.mean(X_jk) * np.mean(X_jkst)
        Cov_st_jkst = np.mean(X_st_jkst) - np.mean(X_st) * np.mean(X_jkst)
        Cov_jk_st = np.mean(X_jkst) - np.mean(X_jk) * np.mean(X_st)

        lambdahat_jkst = Var_jkst + (sigma_st**2)*Var_jk + (sigma_jk**2)*Var_st - 2*sigma_st*Cov_jk_jkst \
                            - 2*sigma_jk*Cov_st_jkst + 2*sigma_jk*sigma_st*Cov_jk_st
#         lambdahat_jkst = Var_jkst
        gamma_jkst = eFC_corr[idx_row[i], idx_col[i]]/math.sqrt(lambdahat_jkst/n)
        gamma_nonzeros[iteration2] = gamma_jkst
    
    return gamma_zeros, gamma_nonzeros




def type1err_subset(check_gamma_zeros, eFC_corr, true_corr_edge, alpha):
    '''
    input:
        gamma: test statistics under FDR procedure (num of selected entries from upper triangular in eFC_corr)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        true_corr_edge: true edge correlation matrix
        alpha: prespecified (edge)x(edge)
    return:
        1/0 for type 1 error each one iteration
        1/0 for type 2 error each one iteration
    '''
#     gamma = np.concatenate((check_gamma_zeros, check_gamma_nonzeros))
    gamma = check_gamma_zeros
    N = len(gamma)
    L = np.sum(1/(np.arange(N) + 1))
    pval = stats.norm.sf(abs(gamma))*2
    
#     pval_sort = np.sort(pval)
#     idx = np.argsort(pval)
#     num_edges = np.shape(true_corr_edge)[1]
#     idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1))
#     r = 0
    
#     for l in (np.arange(N)+1):
#         if pval_sort[l-1] > l*alpha/(N*L):
#             r = l # num of rejections
#             break
            
#     rej_idx = idx[:r] # idx for rejections in pval
#     # first nums of idx_random_zeros corresponds to true H0
#     idx_FP = [i for i in rej_idx if i < len(check_gamma_zeros)]
#     num_FP = len(idx_FP)
    return 



def power_subset(check_gamma_nonzeros, eFC_corr, true_corr_edge, alpha):
    '''
    input:
        gamma: test statistics under FDR procedure (num of selected entries from upper triangular in eFC_corr)
        eFC_corr: estimated edge correlation matrix (edge)x(edge)
        true_corr_edge: true edge correlation matrix
        alpha: prespecified (edge)x(edge)
    return:
        power: num of correct rejections/total num of correct H1
    '''
#     gamma = np.concatenate((check_gamma_zeros, check_gamma_nonzeros))
    gamma = check_gamma_nonzeros
    N = len(gamma)
    L = np.sum(1/(np.arange(N) + 1))
    pval = stats.norm.sf(abs(gamma))*2
    pval_sort = np.sort(pval)
    idx = np.argsort(pval)
    num_edges = np.shape(true_corr_edge)[1]
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_edges, num_edges)), 1))
    r = N
    
    for l in (np.arange(N)+1):
        if pval_sort[l-1] > l*alpha/(N*L):
            r = l # num of rejections
            break
        
#     num_rej = r
            
#     rej_idx = idx[:r] # idx for rejections in pval
#     # nums of idx_random_nonzeros corresponds to true H1
#     idx_TP = [i for i in rej_idx if i >= len(check_gamma_zeros)]
#     num_TP = len(idx_TP)
    return r/len(check_gamma_nonzeros)


def cov2cor(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


