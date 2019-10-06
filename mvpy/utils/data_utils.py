#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:15:38 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
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