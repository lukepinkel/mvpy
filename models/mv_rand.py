#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:04:52 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from numpy import dot, eye, diag, sqrt, zeros
from numpy.linalg import cholesky, inv, multi_dot

from ..utils.linalg_utils import whiten, normalize_diag, sorted_eigh, mdot, chol
from ..utils.base_utils import check_type


def multi_rand(R, size=1000):
    '''
    Generates multivariate random normal matrix
    
    Parameters:
        R: n by n covariance or correlation matrix of the distribution from 
        which the random numbers are to be pulled
        size: size of the random sample
    
    Returns:
        Y: size by n matrix of multivariate random normal values
    '''
    R, col, ix, is_pd = check_type(R)
    
    n = R.shape[0]
    X = whiten(np.random.rand(size, n))
    
    L = chol(R)
    Y = dot(L, X.T).T
    if is_pd:
        Y = pd.DataFrame(Y, columns=col)
    return Y

def random_correlations(n_feats, n_obs=10):
    '''
    Generate random matrix of pearson correlations
    
    Parameters:
        n_feats: number of dimensions for the correlation matrix
        n_obs: number of observations for the random sample matrix
               used to generate the correlation matrix
    
    Returns:
        R: random correlation matrix
    '''
    W = np.random.randn(n_obs, n_feats)
    R = dot(W.T, W) + diag(np.random.rand(n_feats))
    D = diag(1/sqrt(diag(R)))
    R = multi_dot([D, R, D])
    return R

       
def vine_corr(d, betaparams):
    P = zeros((d, d))
    S = eye(d)
    for k in range(d-1):
        for i in range(k+1, d):
            P[k, i] = np.random.beta(betaparams, betaparams)
            P[k, i] = (P[k, i] - 0.5)*2.0
            p = P[k, i]
            for l in range(k-1, 1, -1):
                p = p * sqrt((1 - P[l, i]**2)*(1 - P[l, k]**2)) + P[l, i]*P[l, k]
            S[k, i] = p
            S[i, k] = p
    u, V = sorted_eigh(S)
    umin = np.min(u[u>0])
    u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
    S = mdot([V, diag(u), V.T])
    S = normalize_diag(S)
    return S
        