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
    beta_paths = np.zeros(p)
    for i in range(n_iters):
        for j in range(p):
            active[j] = False
            bj = sft((Xty[~active] - G[~active][:, active].dot(beta[active]))/n, la)
            bj/= dn
            
            beta[j] = bj
            active[j] = True
            
            loglikes.append((penalized_loglike(beta, X, y, lambda_, alpha)))
        beta_paths = np.vstack([beta_paths, beta])        
        llcurr = penalized_loglike(beta, X, y, lambda_, alpha)
        
        if np.abs(llprev - llcurr)<tol:
            break
        if llcurr>llprev:
            break
        llprev = llcurr
        
    return beta, beta_paths, loglikes
    
 
def reorder_gram(G, ix, Cov, Cmax_j, m):
    q = Cmax_j + m
    Cov[[Cmax_j, 0]] = Cov[[0, Cmax_j]]
    ix[q], ix[m] = ix[m], ix[q]
    G[[m, q]] = G[[q, m]]
    G[:, [m, q]] = G[:, [q, m]]
    Cov = Cov[1:]
    return G, Cov, ix
        

  
def get_gamma(Cabsmax, Cov, A, aj):
    if len(Cov)>0:
        Cabsmax = np.atleast_1d(Cabsmax)
        A = np.atleast_1d(A)
        gamprops = np.concatenate([(Cabsmax - Cov) / (A - aj),
                                   (Cabsmax + Cov) / (A + aj),
                                   Cabsmax / A])
        gamprops = gamprops[gamprops>0]
    else:
        gamprops = Cabsmax / A
    gamhat = np.min(gamprops)
    return gamhat


def cho_backsolve(A, b):
    return np.linalg.inv(A.dot(A.T)).dot(b)

def lars(X, y):
    n, p = X.shape
    Cov, Gram = X.T.dot(y), X.T.dot(X)
    L, betas = Gram.copy(), np.zeros((p, p))
    active_set, ix = [], np.arange(p)
    signs = np.zeros(p)
    for i in range(p):
        Cabs = np.abs(Cov)
        Cmax_j = np.argmax(Cabs)
        Cmax, Cabsmax = Cov[Cmax_j], Cabs[Cmax_j]

        signs[i] = np.sign(Cmax)
        Gram, Cov, ix = reorder_gram(Gram, ix, Cov, Cmax_j, i)
        L = linalg_utils.add_chol_row(Gram[i, i], Gram[i, :i], L[:i, :i])
        active_set.append(ix[i])
        j, k = i-1, i+1
        sign_i = signs[:k]
        w = cho_backsolve(L[:k, :k], sign_i)
        A = np.sqrt(1.0 / np.sum(w * sign_i))
        w *= A
        aj = np.dot(Gram[:k, k:].T, w)
        gamhat = get_gamma(Cabsmax, Cov, A, aj)
        betas[i, active_set] = betas[j, active_set] + gamhat * w

        Cov -= gamhat * aj
    return active_set, betas



 

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
       

