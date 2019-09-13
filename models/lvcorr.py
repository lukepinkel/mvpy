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
    if len(pr.shape)>3:
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

def polychor(x, y):
    '''
    Polychoric correlation estimate
    
    Parameters:
        
        x: n by 1 vector
        
        y: n by 1 vector
    
    Returns:
        
        ml_rho: maximum likelihood correlation between x and y
    '''
    X = pd.crosstab(x, y).values
    a, b = thresh(X)
    range1 = np.linspace(-.90, .90, 100)
    ll_step1 =  pd.DataFrame([log_like(X, a, b, z) for z in range1], index=range1)
    llmax1 = ll_step1.idxmax()
    range2 = np.linspace(llmax1-0.1, llmax1+0.1, 50)
    if len(range2.shape)>1:
        range2 = range2[:, 0]
    ll_step2 =  pd.DataFrame([log_like(X, a, b, z) for z in range2], index=range2)
    ml_rho = ll_step2.idxmax()
    return ml_rho

def Polychor(data):
    '''
    Polychoric correlation estimate for a matrix
    
    Parameters:
        data: n by p matrix 
        
    Returns:
        R: maximum likelihood correlation matrix

    '''
    feats = data.columns
    R = [[polychor(data[i], data[j])[0] for c1, i in enumerate(feats)
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