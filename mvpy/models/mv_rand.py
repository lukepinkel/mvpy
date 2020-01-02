#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:04:52 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
from ..utils import linalg_utils, base_utils


def multi_rand(R, size=1000):
    '''
    Generates multivariate random normal matrix.  Not suitable for simulating
    higher order moments, because before being multiplied by the cholesky
    of the specified covariance/correlation matrix, the standard normal
    variates are decorrelated, giving a dataset with a covariance nearly
    exactly equal to that which was specified.
    
    Parameters:
        R: n by n covariance or correlation matrix of the distribution from 
        which the random numbers are to be pulled
        size: size of the random sample
    
    Returns:
        Y: size by n matrix of multivariate random normal values
    '''
    R, col, ix, is_pd = base_utils.check_type(R)
    
    n = R.shape[0]
    X = base_utils.csd(np.random.normal(size=(size, n)))
    X = base_utils.csd(linalg_utils.whiten(X))
    
    W = linalg_utils.chol(R)
    Y = X.dot(W.T)
    if is_pd:
        Y = pd.DataFrame(Y, columns=col)
    return Y

def random_correlations(n_feats, n_obs=10):
    '''
    Generate random matrix of correlations using the factor method
    
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

       
def vine_corr(d, betaparams=10):
    '''
    Generate random matrix of correlations using the vine method
    
    Parameters:
        d: number of dimensions for the correlation matrix
        betaparams: parameter which specifies how intercorrelated the random
                    matrix is to be.  A higher value results in smaller
                    correlations.
    
    Returns:
        R: random correlation matrix
    '''
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
        


def f_moment(params, gamma1, gamma2):
    b, c, d = params
    b2, c2, d2, bd = b**2, c**2, d**2, b * d
    eq1 = b2 + 6.0 * bd + 2.0 * c2 + 15.0 * d2 - 1.0
    eq2 = 2.0 * c * (b2 + 24.0 * bd + 105.0 * d2 + 2) - gamma1
    eq3 = 24.0 * (bd + c2 * (1.0 + b2 + 28.0*bd) 
                  + d2 * (12.0 + 48.0 * bd + 141.0 * c2 + 225.0 * d2)) - gamma2
    return (eq1, eq2, eq3)

def f_corr(r, params, rho):
    if len(params)==8:
        _, b1, c1, d1, _, b2, c2, d2 = params
    else:
        b1, c1, d1, b2, c2, d2 = params       
    eq = r * (b1*b2 + 3*b1*d2 + 3*d1*b2 + 9*d1*d2)
    eq+= r**2 * (2 * c1 * c2)
    eq+= r**3 * (6 * d1 * d2)
    eq-=rho
    return eq

def mcoefs(skew=0, kurtosis=0):
    res = sp.optimize.root(f_moment, (0.1, 0.1, 0.1), args=(skew, kurtosis),
                           method='krylov')
                             
    return res.x, res.fun
    
def mcorr(rho, params):
    res = sp.optimize.root(f_corr, (0.1), args=(params, rho), method='krylov')
                             
    return res.x
    
    
class multivariate_nonnormal:
    
    def __init__(self, mu, cov, skew=None, kurt=None):
        p = len(mu)
        if skew is None:
            skew = np.zeros(p)
        if kurt is None:
            kurt = np.zeros(p)
        self.skew = skew
        self.kurt = kurt
        self.mu = mu
        self.cov = cov
        self.var = np.diag(cov)
        D = np.diag(np.sqrt(1.0/self.var))
        self.corr = D.dot(self.cov).dot(D)
        self._polycoefs = []
        self._polyf = []
        for i in range(p):
            w, t= mcoefs(self.skew[i], self.kurt[i])
            w = np.append(-w[-2], w)
            self._polycoefs.append(w)
            self._polyf.append(t)
        
        self._intermediate_rhovech = []
        for i in range(p):
            for j in range(i):
                wi, wj = self._polycoefs[i], self._polycoefs[j]
                w = np.hstack([wi, wj]).tolist()
                rij = linalg_utils._check_0d(mcorr(self.corr[i, j], w))
                self._intermediate_rhovech.append(rij)
        
        
        self.intermediate_corr = np.eye(p)
        self.intermediate_corr[np.tril_indices(p, -1)] = self._intermediate_rhovech
        self.intermediate_corr[np.triu_indices(p, 1)] = self._intermediate_rhovech
        self.chol_factor = linalg_utils.chol(self.intermediate_corr)
        self.p = p
    def rvs(self, n=1000):
        U = sp.stats.multivariate_normal(np.zeros(self.p), self.intermediate_corr).rvs(n)
        for i in range(self.p):
            xi = U[:, i]
            a, b, c, d = self._polycoefs[i]
            U[:, i] = a + b * xi + c * xi**2 + d * xi**3
        U*=np.sqrt(self.var)
        U+=self.mu
        return U
    
    