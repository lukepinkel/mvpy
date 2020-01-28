#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:41:52 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import scipy.optimize as opt# analysis:ignore

from ..utils import linalg_utils, base_utils # analysis:ignore

import matplotlib.pyplot as plt # analysis:ignore

def sft(x, t):
    y = np.maximum(np.abs(x) - t, 0) * np.sign(x)
    return y

def left_update(S, w2, delta):
    w1_new = sft(S.dot(w2), delta)
    w1_new/= np.linalg.norm(w1_new)
    return w1_new
    

def right_update(S, w1, delta):
    w2_new = sft(w1.dot(S), delta)
    w2_new/= np.linalg.norm(w2_new)
    return w2_new

def lfunc(delta, S, w2, c1):
    if np.max(np.abs(S.dot(w2)) - delta)<=0:
        return -1
    else:
        return np.linalg.norm(left_update(S, w2, delta), 1) - c1
    
def rfunc(delta, S, w1, c2):
    if np.max(np.abs(w1.dot(S)) - delta)<=0:
        return -1
    else:
        return np.linalg.norm(right_update(S, w1, delta), 1) - c2


def weight_update(S, w2, c1, c2):
    delta1, delta2 = 0, 0
    w1_new = left_update(S, w2, delta1)
    if np.linalg.norm(w1_new, 1)>c1:
        delta1 = opt.root_scalar(lfunc, bracket=[0, 50], args=(S, w2, c1), method='brentq')                    
        w1_new = left_update(S, w2, delta1.root)
    w1 = w1_new
    
    w2_new = right_update(S, w1, delta2)
    if np.linalg.norm(w2_new, 1)>c2:
        delta2 = opt.root_scalar(rfunc, bracket=[0, 50], args=(S, w1, c2), method='brentq')                   
        w2_new = right_update(S, w1, delta2.root)
    w2 = w2_new
    return w1, w2
        

def pobjfunc(w1, w2, S, c1, c2):
    ssq = w1.T.dot(S).dot(w2)
    pf = ssq - np.linalg.norm(w1, 1) - np.linalg.norm(w2, 2)
    return pf
    
    
        
        
    
    
def _sparse_cca(X1, X2, c1=None, c2=None, n_iters=50, tol=1e-9):
    if c1 is None:
       c1 = np.sqrt(1 + np.sqrt(X1.shape[1]))
    if c2 is None:
        c2 = np.sqrt(1 + np.sqrt(X1.shape[1]))

    w2 = np.random.normal(0, 1, size=X2.shape[1])
    w2/=np.linalg.norm(w2)
    S = base_utils.cov(X1, X2)
    objfunc_vals = np.zeros((n_iters))
    prev_f = -1e16
    for i in range(n_iters):
        w1_new, w2_new = weight_update(S, w2, c1, c2)
        curr_f = pobjfunc(w1_new, w2_new, S, c1, c2)
        if ((np.abs(curr_f - prev_f)<tol)|(curr_f<prev_f)):
            break 
        prev_f = curr_f
        w1, w2 = w1_new, w2_new
        objfunc_vals[i] = curr_f
    return w1, w2, objfunc_vals[:i]


def sparse_cca(X1, X2, c1=None, c2=None, n_starts=10, n_iters=50, tol=1e-9):
    starts = {'w1':[], 'w2':[], 'fmax':[], 'nit':[], 'r':[], 'p1':[], 'p2':[]}
    S = base_utils.cov(X1, X2)
    for i in range(n_starts):
        w1, w2, fvals = _sparse_cca(X1, X2, c1, c2, n_iters, tol)
        starts['w1'].append(w1)
        starts['w2'].append(w2)
        starts['fmax'].append(fvals[-1])
        starts['nit'].append(len(fvals))
        starts['r'].append(w1.T.dot(S).dot(w2))
        starts['p1'].append(np.sum(np.abs(w1)))
        starts['p2'].append(np.sum(np.abs(w2)))  
    return starts



"""   


S = np.array([[1.0, 0.0, 0.9, 0.0],
              [0.0, 1.0, 0.0, 0.9],
              [0.9, 0.0, 1.0, 0.0],
              [0.0, 0.9, 0.0, 1.0]])
    
w = np.zeros((100, 2)) 
w[:10, 0] = 1.0
w[10:20, 0] = -1.0

w[80:90, 1] = 1.0
w[90:, 1] = -1.0
U = mv.multi_rand(S)



X1 = sp.stats.matrix_normal(U[:, :2].dot(w.T), np.eye(U.shape[0]),
                            np.eye(w.shape[0])).rvs()



X2 = sp.stats.matrix_normal(U[:, 2:].dot(w.T), np.eye(U.shape[0]),
                            np.eye(w.shape[0])).rvs()

 
w1, w2, fvals = _sparse_cca(X1, X2)
    
starts = sparse_cca(X1, X2, c1=4.5, c2=4.5, n_starts=250, n_iters=500, tol=1e-16)
    
start_res = pd.DataFrame([starts[x] for x in ['fmax', 'nit', 'r', 'p1', 'p2']]).T

w1 = starts['w1'][start_res[0].idxmax()]
w2 = starts['w2'][start_res[0].idxmax()]
 

"""





