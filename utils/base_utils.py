#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:39:49 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
from numpy.ma import masked_invalid


def check_type(X):
    if type(X) is pd.DataFrame:
        X,columns,index,is_pd=X.values,X.columns.values,X.index.values,True 
    elif type(X) is pd.Series:
        X, columns, index, is_pd = X.values, X.name, X.index, True
        X = X.reshape(X.shape[0], 1)
    elif type(X) is np.ndarray:
        X, columns, index, is_pd = X, None, None, False 
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
    return X, columns, index, is_pd 



def valid_overlap(X, Y=None):
    if Y is None:
        Y = X
        
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    validX = (1 - np.isnan(X))
    validY = (1 - np.isnan(Y))
    valid = np.dot(validX.T, validY)
    if x_is_pd:
        valid = pd.DataFrame(valid, columns=xcols, index=ycols)
    return valid


def center(X):
    '''
    Centers data
    
    Parameters:
        X: Matrix to be centered
    
    Returns:
        X_centered: Centered X
    '''
    X_centered = X - np.nanmean(X, axis=0)
    return X_centered

def standardize(X):
    '''
    Standardize data
    
    Parameters:
        X: Matrix to be centered
    
    Returns:
        X_standardized: Standardized X
    '''
    X_standardized = X / np.nanstd(X, axis=0)
    return X_standardized


def csd(X):
    '''
    Center Standardize
    
    Parameters:
        X: matrix to be centered and standardized
    Returns:
        y: X minus the mean of X divided by the standard deviation of X
    '''
    return standardize(center(X))

def cov(X, Y=None, bias_corrected=True):
    '''
    Covariance
    
    Parameters:
        X: matrix of n observations by p features for which the covariance is 
           to be calculated
        Y: Optional matrix of n obs by q features. If provided function returns
           the cross covariance (p by x) of X and Y
    Returns:
        S: The covariance matrix of X or cross covariance of X and Y if Y 
           is provided
    ''' 
    X = center(X)
    n = len(X)
    
    if Y is None:
        Y = X
    else:
        Y = center(Y)
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    
   
        
    if np.isnan(X).any().any() is False:
        if bias_corrected:
            n = np.maximum(n-1, 1)
        S = np.einsum('ij,ik->jk', X, Y) / n
        
    else:
        Xm, Ym = masked_invalid(X), masked_invalid(Y)
        if bias_corrected:
            n = valid_overlap(X, Y) - 1
        S = Xm.T.dot(Ym) / n
        S = np.array(S)    
    if ((x_is_pd)|(y_is_pd)):
        S = pd.DataFrame(S, index=xcols, columns=ycols)
    return S


def corr(X, Y=None):
    '''
    Correlation
    
    Parameters:
        X: matrix of n observations by p features for which the correlation is 
           to be calculated
        Y: Optional matrix of n obs by q features. If provided function returns
           the cross correlation (p by x) of X and Y
    Returns:
        S: The correlation matrix of X or cross correlation of X and Y 
           if Y is provided
    '''
    X = csd(X)    
    if Y is None:
        Y = X
    else:
        Y = csd(Y)
        
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    
    R = cov(X, Y)
    
    if ((x_is_pd)|(y_is_pd)):
        R = pd.DataFrame(R, index=xcols, columns=ycols)
        
    return R

