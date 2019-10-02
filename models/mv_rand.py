#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:04:52 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from ..utils import linalg_utils, base_utils


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
    R, col, ix, is_pd = base_utils.check_type(R)
    
    n = R.shape[0]
    X = linalg_utils.whiten(np.random.rand(size, n))
    
    L = linalg_utils.chol(R)
    Y = np.dot(L, X.T).T
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
    R = np.dot(W.T, W) + np.diag(np.random.rand(n_feats))
    D = np.diag(1.0/np.sqrt(np.diag(R)))
    R = np.linalg.multi_dot([D, R, D])
    return R

       
def vine_corr(d, betaparams):
    P = np.zeros((d, d))
    S = np.eye(d)
    for k in range(d-1):
        for i in range(k+1, d):
            P[k, i] = np.random.beta(betaparams, betaparams)
            P[k, i] = (P[k, i] - 0.5)*2.0
            p = P[k, i]
            for l in range(k-1, 1, -1):
                p = p * np.sqrt((1 - P[l, i]**2)*(1 - P[l, k]**2)) + P[l, i]*P[l, k]
            S[k, i] = p
            S[i, k] = p
    u, V = linalg_utils.sorted_eigh(S)
    umin = np.min(u[u>0])
    u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
    S = linalg_utils.mdot([V, np.diag(u), V.T])
    S = linalg_utils.normalize_diag(S)
    return S

 
def onion_corr(d, betaparams=5):
    S = np.eye(d)
    S[0, 1] = 2*np.random.beta(betaparams, betaparams)-1.0
    S[1, 0] = S[0, 1]
    r = S[:2, :2]
    for k in range(2, d-1):
        y = np.sqrt(np.random.beta(k/2.0, betaparams))
        u = np.random.rand(k)
        u /= np.linalg.norm(u)
        u *= y
        A = np.linalg.cholesky(r)
        z = A.dot(u)[:, None]
        r = np.block([[r, z], [z.T, np.ones((1, 1))]])
    S = r
    u, V = linalg_utils.sorted_eigh(S)
    umin = np.min(u[u>0])
    u[u<0] = [umin*0.5**(float(i+1)/len(u[u<0])) for i in range(len(u[u<0]))]
    S = linalg_utils.mdot([V, np.diag(u), V.T])
    S = linalg_utils.normalize_diag(S)
    return S
        