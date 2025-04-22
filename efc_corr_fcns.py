#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets
import matplotlib.pyplot as plt
import itertools
import math
import random
from scipy.linalg import sqrtm
from scipy.linalg import block_diag




# calculate edge time series from data
def fcn_ets(x):
    '''
    input:
        x: observed BOLD signals time series (time)x(node)
    return: 
        edge time series (time)x(edge)
    
    '''
    # zscore
    x_zs = stats.zscore(x, axis=0, ddof=1)
    num_nodes = np.shape(x_zs)[1]
    # indicies for edges
    idx_row, idx_col = np.nonzero(np.triu(np.ones((num_nodes, num_nodes)), 1))
    edge_mat = np.multiply(x_zs[:,idx_row], x_zs[:,idx_col])
    return edge_mat


# eFC

def fcn_eFC(ets):
    '''
    input:
        ets: edge time series 2d array (time)x(edge)
    return:
        eFC_mat: eFC matrix (edge)x(edge)
    '''
    
    # inter product
    product = np.dot(ets.T, ets)
    # scaling
    sd = np.sqrt(product.diagonal()).reshape(-1,1)
    eFC_mat = product/(np.outer(sd, sd))
    return eFC_mat



# sample moment estimator
# calculate edge correlation
def fcn_eFC_corr(ets):
    '''
    estimate E(X1X2X3X4)-E(X1X2)E(X3X4)
    input:
        ets: edge time series 2d array (time)x(edge)
    return:
        eFC_corr_mat: edge correlation matrix (edge)x(edge)
    '''
    # center edge time series
    ets_center = ets - ets.mean(axis=0)
    # inter product
    product = np.dot(ets_center.T, ets_center)
    # scaling
    sd = np.sqrt(product.diagonal()).reshape(-1,1)
    eFC_corr_mat = product / (np.outer(sd, sd))
    return eFC_corr_mat



# functions for true correlation matrix of edges

def central_second_moment(cov,i,j):
    return cov[i,j]


def central_fourth_moment(cov,mean,i,j,k,n):
    '''
    input:
        cov: covariance matrix
        i,j,k,n: four variables Xi, Xj, Xk, Xn (starting from 0)
    return:
        central fourth moment E[XiXjXkXn]    
    '''
    moment = cov[i,j]*cov[k,n] + cov[i,k]*cov[j,n] + cov[i,n]*cov[j,k]
    return moment


def corr_edge(cov, mean):
    '''
    input:
        cov: true covariance matrix
        mean: true mean vector
    return:
        corr_edge: true correlation matrix of edges
    '''
    n = mean.shape[0] # number of variables (nodes)
    a = list(itertools.combinations(np.arange(n), 2)) # all combinations of XiXj
    b = np.array(list(itertools.combinations(np.arange(len(a)), 2))) # all combinations of index in a
    cov_edge = np.zeros((len(a), len(a))) # covariance matrix of edges
    
    # off-diagonal
    for p in b:
        i = a[p[0]][0]
        j = a[p[0]][1]
        k = a[p[1]][0]
        n = a[p[1]][1]
        cov_edge[p[0], p[1]] = central_fourth_moment(cov,mean,i,j,k,n) - central_second_moment(cov,i,j)*central_second_moment(cov,k,n)
    
    # diagonal
    for i in np.arange(len(a)):
        k = a[i][0]
        n = a[i][1]
        cov_edge[i,i] = central_fourth_moment(cov,mean,k,n,k,n) - central_second_moment(cov,k,n)*central_second_moment(cov,k,n)
    cov_edge = np.triu(cov_edge,1) + cov_edge.T
    sd = np.sqrt(np.diag(cov_edge))
    corr_edge = cov_edge / (np.outer(sd, sd))
    corr_edge[cov_edge == 0] = 0
    return corr_edge



def cov_to_corr(cov):
    '''
    convert covariance matrix to correlation
    '''
    sd = np.sqrt(np.diag(cov))
    corr = cov / (np.outer(sd, sd))
    corr[cov == 0] = 0
    return corr



