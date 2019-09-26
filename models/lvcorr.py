#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:55:34 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from numpy import dot, exp, log
from numpy.linalg import inv
from scipy.optimize import minimize

from ..utils.base_utils import check_type, corr
from ..utils.statfunc_utils import norm_qtf, binorm_cdf, polyserial_ll
def tetra(X, return_sterr=False):
    '''
    Tetrachoric correlations inferred from binary data
    
    Parameters:
        
        X: n observation by p features matrix
        return_sterr: Return standard error of the estimates
    Returns:
        R: p by p correlation matrix
        Rl: (optional) lower 95% CI for estimate
        Ru: (optional) upper 95% CI for estimate
    '''
    X, cols, ix, is_pd = check_type(X)
    A = dot(X.T, X) + 0.5
    B = dot((1 - X).T, X) + 0.5
    C = dot(X.T, (1 - X)) + 0.5
    D = dot((1 - X).T, (1 - X))
    
    
    delta = (A * D) / ((B * C))
    gamma = 1 + delta**.5
    
    R = np.cos(np.pi/gamma) 
    
    if is_pd is True:
        R = pd.DataFrame(R, index=cols, columns=cols)
    if return_sterr is False:
        return R
    else:
        SE = (inv(A) + inv(B) + inv(C) + inv(D))**0.5
        delta_l, delta_u = delta/exp(SE), delta*exp(SE)
        Rl, Ru = np.cos(np.pi/(1+delta_l**.5)), np.cos(np.pi/(1+delta_u**.5))
        return R, Rl, Ru
    
 

def thresh(X):
    '''
    Maximum likelihood estimates for thresholds
    
    Parameters:
        X: crosstabulation table
    Returns:
        a: thresholds for axis 0
        b: thresholds for axis 1
    '''
    N = float(np.sum(X))
    a = norm_qtf(np.sum(X, axis=0).cumsum() / N)[:-1]
    b = norm_qtf(np.sum(X, axis=1).cumsum() / N)[:-1]
    a, b = np.concatenate([[-1e6], a, [1e6]]), np.concatenate([[-1e6], b, [1e6]])
    return a, b

def probs(a, b, r):
    '''
    Cumulative bivariate normal distribution.  Computes the probability
    that a value falls in category i,j
    
    Parameters:
        a: Thresholds along axis 0
        b: Thresholds along axis 1
        r: correlation coefficient
    
    Returns:
        pr: Matrix of probabilities
    '''
    pr = np.array([[binorm_cdf(x, y, r) for x in a] for y in b])
    return pr

def log_like(X, a, b, r):
    '''
    Log likelihood of a contingency table given thresholds and  the correlation
    coefficient
    
    Parameters:
        X: Contigency table
        a: Thresholds along axis 0
        b: Thresholds along axis 1
        r: correlation coefficient
    Returns:
        ll: Log likelihood
    '''
    pr = probs(a, b, r)
    if len(pr.shape)>=3:
        pr = pr[:, :, 0]
    n, k = pr.shape
    pr = np.array([[pr[i, j]+pr[i-1,j-1]-pr[i-1,j]-pr[i,j-1] 
                   for j in range(1,k)] for i in range(1,n)])
    ll = np.sum(X * log(pr))
    return ll

def normal_categorical(x, nx):
    '''
    Splits continuous variable into nx categories
    
    Parameters:
        x: continuous vaiable in an array
        nx: number of categories
    
    Returns:
        xcat: categorical x
    '''
    xcat = pd.qcut(x, nx, labels=[i for i in range(nx)]).astype(float)
    return xcat


def polychor_ll(params, X, k):
    rho = params[0]
    a, b = params[1:k+1], params[k+1:]
    return -log_like(X, a, b, rho)



def polychor_partial_ll(rho, X, k, params):
    a, b = params[:k], params[k:]
    return -log_like(X, a, b, rho)


def polychorr(x, y, ret_optimizer=False):
    xtab = pd.crosstab(x, y).values
    a, b = thresh(xtab)
    k = len(a)
    rinit =  np.corrcoef(x, y, rowvar=False)
    
    params = np.concatenate([rinit, a, b], axis=0)
    ca =[dict(zip(['type', 'fun'], ['ineq', lambda params: params[i+1]-params[i]]))
         for i in range(1, k+1)]
    
    cb = [dict(zip(['type', 'fun'], ['ineq', lambda params: params[i+1]-params[i]]))
          for i in range(k, k+len(b))]
    constr= ca+cb
    
    bounds = [(-1.0, 1.0)]+ [(None, None) for i in range(len(params)-1)]
    
    optimizer = minimize(polychor_ll, params, args=(xtab, k), bounds=bounds,
                   constraints=constr)
    
    if np.isnan(optimizer.fun):
        optimizer = minimize(polychor_ll, params, args=(xtab, k), bounds=bounds)
        
    params = optimizer.x
    rho, a, b = params[0], params[1:k+1], params[k+1:]
    if ret_optimizer is False:
        return rho, a, b
    else:
        return rho, a, b, optimizer


def Polychorr(data):
    '''
    Polychoric correlation estimate for a matrix
    
    Parameters:
        data: n by p matrix 
        
    Returns:
        R: maximum likelihood correlation matrix

    '''
    feats = data.columns
    R = [[polychorr(data[i], data[j])[0] for c1, i in enumerate(feats)
    if c1<c2] for c2, j in enumerate(feats)]      
    R = pd.DataFrame(R, index=feats, columns=feats[:-1])
    R[feats[-1]] = [np.nan for i in range(len(feats))]
    for i in feats:
        for c in feats:
            if i==c:
                R.loc[i, c]=1
            else:
                R.loc[i, c] = R.loc[c, i]
    return R



def polyserial(x, y):
    '''
    Polyserial correlation.  Estimates the correlation coefficient
    between a categorical and continuous variable, under the assumption
    that the continuous variable is an arbitrarily thresholded normally
    distributed variable
    
    Parameters:
        x: Continuous variable
        y: Categorical variable
        
    Returns:
        rho_hat: Estimated correlation 
    '''
    
    order = dict(zip(np.unique(y), np.unique(y).argsort()))

    marginal_counts = np.array([np.sum(y==z) for z in np.unique(y)]).astype(float)
    tau = norm_qtf(marginal_counts.cumsum()/marginal_counts.sum())
    tau = np.concatenate([[-np.inf], tau])
    
        
    res = minimize(polyserial_ll, x0=(corr(x, y)), args=(x, y, tau, order),
                   method='Nelder-Mead')
    rho_hat=res.x
    return rho_hat