#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:15:38 2019

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd
from .base_utils import check_type
        
def dummy_encode(X, colnames=None, complete=False):
    '''
    Dummy encodes a categorical variable
    
    Parameters:
        X: n by one matrix of categories
        colnames: Labels for categories
        complete: Whether or not to encode each category as a column, in a 
                  redundant fashion
    '''
    X, cols, ix, is_pd = check_type(X)
    cats = np.unique(X)
    n_cats = len(cats)
    if complete is False:
        dummy_vars = [(X==cats[i]) for i in range(n_cats-1)]
    else:
        dummy_vars = [(X==cats[i]) for i in range(n_cats)]
    dummy_vars = np.concatenate(dummy_vars, axis=1) * 1.0
    
    if is_pd is True:
        if colnames is not None:
            cats = colnames
        elif complete is False:
            cats = cats[:-1]
        else:
            cats = cats
        dummy_vars = pd.DataFrame(dummy_vars, columns=cats, index=ix)
    return dummy_vars




def xcorr_fftc(x, y, normalization="coef", retlags=True):
    n = x.shape[0]
    rho = np.zeros(2*n-1)
    r = sp.signal.fftconvolve(x, y[::-1], mode='full')[n-1:]
    rho[n-1:] = r
    rho[:n-1] = r[1:][::-1]
    lags = np.arange(1-n, n)
    if normalization=='unbiased':
        c = 1.0 / (n - np.abs(lags))
    elif normalization=='biased':
        c = 1.0 / n
    elif normalization=='coef':
        c = 1.0 / np.sqrt(np.dot(x, x)*np.dot(y, y))
    rho*=c
    if retlags:
        return lags, rho
    else:
        return rho