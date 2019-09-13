#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:14:31 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd

from numpy import log, exp, sqrt, eye, dot, trace, pi
from numpy.linalg import slogdet, pinv
from scipy.special import erf, erfinv
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize

from .base_utils import corr, check_type

def norm_pdf(x, mu=0, s=1):
    '''
    Normal probability distribution function
    
    Parameters:
        x: A number, vector or matrix
        mu: default 0, the mean of the gaussian pdf
        s: default 1, the standard deviation
    
    Returns:
        y: The density of the pdf evaluated at x
    '''
    c = 1 / sqrt(2*pi*s**s)
    y = c * exp(-(x-mu)**2 / (2*s**2))
    return y

def norm_cdf(x, mu=0, s=1):
    '''
    Normal cumulative probability distribution function
    
    Parameters:
        x: Number, vector or matrix
        mu: default 0, the mean
        s: default 1, the standard deviation
    
    Returns:
        y: cumulative distribution from negative infinity to x
    '''
    y = 0.5 * (1 + erf((x-mu) / (sqrt(2)*s)))
    return y

def norm_qtf(x):
    '''
    Normal quantile function - the inverse of the cdf
    
    Parameters:
        x: Number, vector or matrix
    
    Returns:
        prob: The value or values that would produce the input for a standard
              normal cdf
    '''
    prob = sqrt(2) * erfinv(2 * x - 1)
    return prob

def binorm_pdf(x, y, r, mu_x=0, mu_y=0, sx=1, sy=1):
    '''
    Bivariate normal probability distribution function
    
    Parameters:
        x: Number, vector or matrix to be evaluated
        y: Number, vector or matrix to be evaluated
        r: Correlation between x and y
        mu_x: default 0, the mean of x
        mu_y: default 0, the mean y
        sx: default 1, the standard deviation of x
        sy: default 1, the standard deviation of y
    '''
    r2 = (1 - r**2)
    c0 = 1 / (2 * pi *sx * sy * sqrt(r2))
    c1 = -1/(2 * r2)
    eq1 = ((x - mu_x)**2) / (sx**2)
    eq2 = ((y - mu_y)**2) / (sy**2)
    eq3 = (2 * r * (x - mu_x) * (y - mu_y)) / (sx * sy)
    p = c0 * exp(c1 * (eq1 + eq2 - eq3))
    return p

def binorm_dl(h, k, r):
    '''
    Derivative of the bivariate normal distribution with respect to rho

    '''
    r2 = 1 - r**2
    constant = 1 / (2 * pi * sqrt(r2))
    dl = exp(-(h**2-2*r*h*k+k**2) / (2 * r2))
    dldp = dl * constant
    return dldp

def binorm_l(h, k, r):
    '''
    Bivariate normal likelihood function
    '''
    eq1 = 5 * binorm_dl(h, k, (1 - sqrt(3/5)) * r / 2)
    eq2 = 8 * binorm_dl(h, k, r/2)
    eq3 = 5 * binorm_dl(h, k, (1 + sqrt(3/5)) * r / 2)
    likelihood = r * (eq1 + eq2 + eq3) / 18
    likelihood += norm_cdf(-h) * norm_cdf(-k)
    return likelihood

def binorm_cdf(h, k, r):
    '''
    Approximation of the bivariate normal cumulative distribution using
    chebyshev polynomials
    '''
    likelihood = binorm_l(h, k, r)
    phi = likelihood + norm_cdf(h) + norm_cdf(k) - 1
    return phi


def srmr(Sigma, S, df):
    p = S.shape[0]
    y = 0.0
    t = (p + 1.0) * p
    for i in range(p):
        for j in range(i):
            y += (Sigma[i, j]-S[i, j])**2/(S[i, i]*S[j, j])
    
    y = sqrt((2.0 / (t)) * y)      
    return y

def lr_test(Sigma, S, df):
    p = Sigma.shape[0]
    chi2 = slogdet(Sigma)[1] + trace(dot(pinv(Sigma), S)) - slogdet(S)[1] - p
    pval = 1.0 - chi2_dist.cdf(chi2, (p + 1)*p/2)
    return chi2, pval

def gfi(Sigma, S):
    p = S.shape[0]
    tmp1 = pinv(Sigma).dot(S)
    tmp2 = tmp1 - eye(p)
    y = 1.0 - trace(dot(tmp2, tmp2)) / trace(dot(tmp1, tmp1))
    return y

def agfi(Sigma, S, df):
    p = S.shape[0]
    t = (p + 1.0) * p
    tmp1 = pinv(Sigma).dot(S)
    tmp2 = tmp1 - eye(p)
    y = 1.0 - trace(dot(tmp2, tmp2)) / trace(dot(tmp1, tmp1))
    y = 1.0 - (t / (2.0*df)) * (1.0-y)
    return y




def polyex(x, tau, rho):
    return (tau - rho*x) / sqrt(1-rho**2)

def polyserial_ll(rho, x, y, tau, order):
    ll = []
    for xi, yi in list(zip(x, y)):
        k = order[yi]
        tau1, tau2 = polyex(xi, tau[k+1], rho), polyex(xi, tau[k], rho)
        ll.append(log(norm_cdf(tau1)-norm_cdf(tau2)))
    ll = -np.sum(np.array(ll), axis=0)
    return ll

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

  
def empirical_cdf(X):
    '''
    Empirical cumulative distribution function
    
    Parameters:
        X: Array
    
    Returns:
        y: values of the empirical cdf
    '''
    if type(X) is pd.DataFrame:
        X = X.values
    if len(X.shape)>1:
        X = X[:, 0]
    n = len(X)
    idx = X.argsort()
    y = pd.DataFrame(np.arange(1, n+1)/float(n), index=X[idx])
    return y


def fdr_bh(p_values):
    p_values, cols, ix, is_pd = check_type(p_values)
    idx = np.argsort(p_values)
    correction = empirical_cdf(p_values[idx])
    p_values[idx] /= correction
    
    if is_pd:
        if type(cols) is str:
            cols = list(cols)
        p_values = pd.DataFrame(p_values, columns=cols, index=ix)
    return p_values
    