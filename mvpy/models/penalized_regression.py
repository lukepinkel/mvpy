#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:25:48 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
from ..utils import linalg_utils, base_utils # analysis:ignore
from .glm3 import Binomial
def sft(x, t):
    y = np.maximum(np.abs(x) - t, 0) * np.sign(x)
    return y
    
    
def penalty_term(beta, lambda_=0.1, alpha=0.5):
    p = np.sum(0.5 * (1 - alpha) * beta**2 + alpha*np.abs(beta))
    return p

def working_variate(f, eta, y):
    mu = f.inv_link(eta)
    w = f.dlink(mu) # f.var_func(mu=mu)
    z = eta + (y - mu) / w
    return z, w
    
def weighted_update(X, X2, w, y, yhat, la, dn):
    w = linalg_utils._check_2d(w)
    Xw = X * w
    numerator = sft(Xw.T.dot(y - yhat), la)
    denominat = np.sum(X2 * w, axis=0)+ dn
    return numerator / denominat

def weighted_penalized_loglike(beta, X, w, z, lambda_, alpha):
    w = linalg_utils._check_2d(w)
    r = linalg_utils._check_2d(z - X.dot(beta))
    ssq = np.dot((r * w).T, r) / X.shape[0]
    pll  =ssq + penalty_term(beta, lambda_, alpha)
    return pll
    
def penalized_loglike(beta, X, y, lambda_=0.1, alpha=0.5):
    r = y - X.dot(beta)
    ssr = np.dot(r.T, r) / (2 * X.shape[0])
    pll = ssr + penalty_term(beta, lambda_, alpha)
    return pll
    
def eln_coordinate_descent(X, y, lambda_=0.1, alpha=0.5, n_iters=20, tol=1e-9):
    X, y = base_utils.csd(X), base_utils.csd(y)
    G, Xty = X.T.dot(X),  X.T.dot(y)
    beta = np.linalg.inv(G).dot(Xty)
    n, p = X.shape
    active = np.ones(p).astype(bool)
    la, dn = lambda_ * alpha, lambda_ * (1  - alpha) + 1.0
    loglikes, llprev = [], penalized_loglike(beta, X, y, lambda_, alpha)
    
    for i in range(n_iters):
        for j in range(p):
            active[j] = False
            bj = sft((Xty[~active] - G[~active][:, active].dot(beta[active]))/n, la)
            bj/= dn
            
            beta[j] = bj
            active[j] = True
            
            loglikes.append((penalized_loglike(beta, X, y, lambda_, alpha)))
            
        llcurr = penalized_loglike(beta, X, y, lambda_, alpha)
        
        if np.abs(llprev - llcurr)<tol:
            break
        
        llprev = llcurr
        
    return beta, loglikes
    
    

def penalized_glm_cd(X, y, f=None, lambda_=0.1, alpha=0.5, n_iters=20,
                     tol=1e-4, vocal=False):
    if f is None:
        f = Binomial()
    X = base_utils.csd(X)
    n, p = X.shape
    active = np.ones(p).astype(bool)
    beta = np.linalg.pinv(X).dot(y)
    la, dn = lambda_ * alpha, lambda_ * (1  - alpha) + 1.0
    loglikes, llprev = [], 1e16
    X2 = X**2
    
    for i in range(n_iters):
        
        for j in range(p):
            
            active[j] = False
            eta = X[:, active].dot(beta[active])
            z, w = working_variate(f, eta, y)
            bj  = weighted_update(X, X2, w, z, eta, la, dn)[~active]
            beta[j] = bj
            active[j] = True
            
            if vocal:
                print(i, j)
                
            llcurr = weighted_penalized_loglike(beta, X, w, z, lambda_, alpha)
            
            if np.abs(llprev - llcurr)<tol: 
                break
            
            llprev = llcurr
            loglikes.append(llcurr)
            
    return loglikes, beta
       

