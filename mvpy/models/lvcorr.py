#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:55:34 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize

from ..utils.linalg_utils import _check_1d
from ..utils.base_utils import check_type, corr
from ..utils.statfunc_utils import (norm_qtf, polyserial_ll,
                                    polychor_thresh, polychor_ll, polychor_partial_ll)

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
    A = np.dot(X.T, X) + 0.5
    B = np.dot((1 - X).T, X) + 0.5
    C = np.dot(X.T, (1 - X)) + 0.5
    D = np.dot((1 - X).T, (1 - X))
    
    
    delta = (A * D) / ((B * C))
    gamma = 1 + delta**.5
    
    R = np.cos(np.pi/gamma) 
    
    if is_pd is True:
        R = pd.DataFrame(R, index=cols, columns=cols)
    if return_sterr is False:
        return R
    else:
        SE = (np.linalg.inv(A) + np.linalg.inv(B)\
              + np.linalg.inv(C) + np.linalg.inv(D))**0.5
        delta_l, delta_u = delta/np.exp(SE), delta*np.exp(SE)
        Rl, Ru = np.cos(np.pi/(1+delta_l**.5)), np.cos(np.pi/(1+delta_u**.5))
        return R, Rl, Ru
    
 
def polychorr(x, y, ret_optimizer=False, method=2):
    x, xcols, xix, x_is_pd = check_type(x)
    y, ycols, yix, y_is_pd = check_type(y)
    x, y = pd.Series(_check_1d(x)), pd.Series(_check_1d(y))
    xtab = pd.crosstab(x, y).values
    a, b = polychor_thresh(xtab)
    k = len(a)
    rinit = np.array([np.corrcoef(x, y, rowvar=False)[0, 1]])
    
    params = np.concatenate([rinit, a, b], axis=0)
    ca =[dict(zip(['type', 'fun'], ['ineq', lambda params: params[i+1]-params[i]]))
         for i in range(1, k+1)]
    
    cb = [dict(zip(['type', 'fun'], ['ineq', lambda params: params[i+1]-params[i]]))
          for i in range(k, k+len(b))]
    constr= ca+cb
    
    bounds = [(-1.0, 1.0)]+ [(None, None) for i in range(len(params)-1)]
    
    if method == 1:
        optimizer = sp.optimize.minimize(polychor_ll, params, args=(xtab, k), bounds=bounds,
                   constraints=constr)
    
    elif method == 2:
        optimizer = sp.optimize.minimize(polychor_ll, params, args=(xtab, k), bounds=bounds)
    elif method == 3:
         optimizer = sp.optimize.minimize(polychor_partial_ll, rinit, 
                                          args=(xtab, k, params[1:]), 
                                          bounds=[bounds[0]])
    params = optimizer.x
    if method !=3:
        rho, a, b = params[0], params[1:k+1], params[k+1:]
    else:
        rho, a, b = params[0], params[:k], params[k:]
    if ret_optimizer is False:
        return rho, a, b
    else:
        return rho, a, b, optimizer



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
    x, xcols, xix, x_is_pd = check_type(x)
    y, ycols, yix, y_is_pd = check_type(y)
    
    order = dict(zip(np.unique(y), np.unique(y).argsort()))

    marginal_counts = np.array([np.sum(y==z) for z in np.unique(y)]).astype(float)
    tau = norm_qtf(marginal_counts.cumsum()/marginal_counts.sum())
    tau = np.concatenate([[-np.inf], tau])
    
        
    res = sp.optimize.minimize(polyserial_ll, x0=(corr(x, y)), args=(x, y, tau, order),
                   method='Nelder-Mead')
    rho_hat=res.x
    if (x_is_pd | y_is_pd):
        if type(xcols) is not str:
            xcols = xcols[0]
        if type(ycols) is not str:
            ycols = ycols[0]
        rho_hat = pd.DataFrame([[rho_hat]], index=[ycols],
                               columns=[xcols])
    return rho_hat



def _infer_ctype(vi, vj):
    if (vi=="Contin")&(vj=="Contin"):
        return "corr"
    elif (((vi=="Contin")&(vj=="Ord"))|((vi=="Ord")&(vj=="Contin"))):
        return "polyserial"
    elif (vi=="Ord")&(vj=="Ord"):
        return "polychorr"
    


def mixed_corr(data, var_types=None):
    X, xcols, xix, is_pd = check_type(data)
    if var_types in ["infer", None]:
        var_types = [np.unique(X[:, i]) for i in range(X.shape[1])]
        var_types = ["Ord" if v<7 else "Contin" for v in var_types]
    elif type(var_types) is dict:
        var_types = list(var_types.values())
    
    R = np.zeros((data.shape[1], data.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            if i==j:
                R[i, j] = 0.5
            else:
                ctype = _infer_ctype(var_types[i], var_types[j])
                if ctype=="corr":
                    R[i, j] = corr(X.iloc[:, i], X.iloc[:, j])
                elif ctype=="polchorr":
                    R[i, j] = polychorr(X.iloc[:, i], X.iloc[:, j])[0]
                elif ctype=="polserial":
                    R[i, j] = polychorr(X.iloc[:, i], X.iloc[:, j])
    R += R.T
    if is_pd:
        R = pd.DataFrame(R, index=xcols, columns=xcols)
    return R
            
                
       






