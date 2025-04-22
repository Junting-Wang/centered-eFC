#!/usr/bin/env python

import scipy
from scipy import stats
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from efc_corr_fcns import *

def Frobenius_norm(mat):
    row = mat.shape[0]
    col = mat.shape[1]
    sum_sq = 0.0
    for i in np.arange(row):
        for j in np.arange(col):
            sum_sq += mat[i,j]**2
    return math.sqrt(sum_sq)

    
def max_norm(mat):
    return np.absolute(mat).max()

    
def combination(n, k):
    return math.factorial(n)/(math.factorial(n-k) * math.factorial(k))


# Sample moment estimator
def simulation_efc_corr(seed, mean, cov, true_corr_edge, n):
    '''
    input:
        rep: number of repetitions
        mean: mean vector
        cov: covariance matrix
        n: number of samples
    return:
        fnorms: Frobenuis norms of eFC_corr errors (rep)x(2)
    '''
    fnorms = 0.0
    
    p = len(mean)
    comb = int(combination(p, 2)) # number of combinations
    sum_eFC_corr = np.zeros([comb, comb])
    
    np.random.seed(seed)
    # generate dataa
    X = np.random.multivariate_normal(mean, cov, n)

    # calculate correlation matrices
    ets = fcn_ets(X)
    eFC_corr = fcn_eFC_corr(ets)
#         true_corr_edge = corr_edge(cov, mean)

    # get matrices of errors
    eFC_corr_err = eFC_corr - true_corr_edge

    # Frobenius norms of error matrices
    fnorms = Frobenius_norm(eFC_corr_err)

    return fnorms


# Regularized estimator
import copy
def simulation_efc_corr_reg(seed, mean, cov, true_corr_edge, n, Cs):
    '''
    input:
        rep: number of repetitions
        mean: mean vector
        cov: covariance matrix
        n: number of samples
    return:
        fnorms: Frobenuis norms of eFC_corr errors (rep)x(2)     
    '''
    fnorms = []
    
    p = len(mean)
    
    np.random.seed(seed)
    # generate data
    X = np.random.multivariate_normal(mean, cov, n)

    # calculate correlation matrices
    ets = fcn_ets(X)
    eFC_corr = fcn_eFC_corr(ets)
    eFC_corr_thold = copy.deepcopy(eFC_corr)
    
    for C in Cs:
        t = math.sqrt((math.log(n*(p**4))**4) / n)*C
        eFC_corr_thold[np.absolute(eFC_corr_thold) < t] = 0

        # get matrices of errors
        eFC_corr_thold_err = eFC_corr_thold - true_corr_edge

        # Frobenius norms of error matrices
        fnorms.append(Frobenius_norm(eFC_corr_thold_err))

    return fnorms


def TFPR(mat, true_mat):
    '''
    input: 
        mat: estimated matrix
        true_mat: true matrix
    return:
        TPR, FPR: true positive rate, false positive rate
    '''
    TP = np.sum(np.logical_and(mat != 0, true_mat != 0))
    FN = np.sum(np.logical_and(mat == 0, true_mat != 0))
    FP = np.sum(np.logical_and(mat != 0, true_mat == 0))
    TN = np.sum(np.logical_and(mat == 0, true_mat == 0))
    
    TPR = TP/(TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR



def threshold_simulation(rep, mean, cov, n, reverseC):
    '''
    input:
        rep: number of repetitions
        mean: mean vector
        cov: covariance matrix
        n: number of samples
        reverseC: threshold constant, 1/C
    return:
        fnorms: Frobenuis norms mean of eFC_corr, eFC_corr_threshold errors over repetitions (1)x(2)
        TPRs: true positive rate of eFC_corr_threshold over repetitions
        FPRs: true positive rate of eFC_corr_threshold over repetitions
    '''
    p = len(mean)
    t = math.sqrt((math.log(n*(p**4))**4) / n)/reverseC
    fnorms = np.empty([rep, 2])
    TPRs = np.empty([rep, 1])
    FPRs = np.empty([rep, 1])
#     Cs = list(range(0, C+1, 50))
#     Cs[0] = 1
#     roc = np.zeros([len(Cs),2])
    
    for i in np.arange(rep):
        np.random.seed(10*i)
        # generate dataa
        X = np.random.multivariate_normal(mean, cov, n)
    
        # calculate correlation matrices
        ets = fcn_ets(X)
#         eFC = fcn_eFC(ets)
        temp = fcn_eFC_corr(ets)
        eFC_corr = temp
#         eFC_corr_thold = fcn_eFC_corr(ets)
        eFC_corr_thold = copy.deepcopy(temp)
        eFC_corr_thold[np.absolute(eFC_corr_thold) < t] = 0
        true_corr_edge = corr_edge(cov, mean)
    
        # get matrices of errors
        eFC_corr_err = eFC_corr - true_corr_edge
        eFC_corr_thold_err = eFC_corr_thold - true_corr_edge
    
        # Frobenius norms of error matrices
        fnorms[i, 0] = Frobenius_norm(eFC_corr_err)
        fnorms[i, 1] = Frobenius_norm(eFC_corr_thold_err)
        
#         TPRs[i,0], FPRs[i,0] = TFPR(eFC_corr, true_corr_edge)
        TPRs[i,0], FPRs[i,0] = TFPR(eFC_corr_thold, true_corr_edge)

    return np.mean(fnorms, axis=0), np.mean(TPRs, axis=0), np.mean(FPRs, axis=0)



def threshold_roc(rep, mean, cov, true_corr_edge, n, reverseC):
    '''
    input:
        rep: number of repetitions
        mean: mean vector
        cov: covariance matrix
        n: number of samples
        reverseC: threshold constant, 1/C
    return:
        roc: 2d array: 1st col: TPR, 2nd col: FPR
    '''
    p = len(mean)
    Cs = list(range(0, reverseC+1, 50))
    Cs[0] = 1
    roc = np.zeros([len(Cs),2])
    
    for i in np.arange(rep):
        np.random.seed(10*i)
        # generate dataa
        X = np.random.multivariate_normal(mean, cov, n)
    
        # calculate correlation matrices
        ets = fcn_ets(X)
        temp = fcn_eFC_corr(ets)
#         true_corr_edge = corr_edge(cov, mean)
        
        # create roc curves based on different C
        for j in np.arange(len(Cs)):
            t = math.sqrt((math.log(n*(p**4))**4) / n)/Cs[j]
            eFC_corr_thold = copy.deepcopy(temp)
            eFC_corr_thold[np.absolute(eFC_corr_thold) < t] = 0
            TPR, FPR = TFPR(eFC_corr_thold, true_corr_edge)
            roc[j,0] = roc[j,0] + TPR
            roc[j,1] = roc[j,1] + FPR

    return roc/rep
    