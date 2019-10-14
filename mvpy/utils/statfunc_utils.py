#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:14:31 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import numpy.ma
from numpy import log, exp, sqrt, eye, dot, trace, pi
from numpy.linalg import slogdet, pinv
from scipy.special import erf, erfinv
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize #analysis:ignore

from .base_utils import corr, check_type, valid_overlap #analysis:ignore
from .linalg_utils import _check_1d, _check_np

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
    root1 = sqrt(5.0 - 2 * sqrt(10.0/7.0)) / 3
    root2 = sqrt(5.0 + 2 * sqrt(10.0/7.0)) / 3
    r2 = r/2.0
    w1 = 128.0/225.0
    w2 = (322.0+13.0*sqrt(70.0)) / 900.0
    w3 = (322.0-13.0*sqrt(70.0)) / 900.0
    
    eq1 = w1 * binorm_dl(h, k, r / 2)
    
    eq2 = w2 * binorm_dl(h, k, (1-root1) * r2)
    eq3 = w2 * binorm_dl(h, k, (1+root1) * r2)
    
    eq4 = w3 * binorm_dl(h, k, (1-root2) * r2)
    eq5 = w3 * binorm_dl(h, k, (1+root2) * r2)

    likelihood = r2 * (eq1 + eq2 + eq3 + eq4 + eq5) 
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
    S = _check_np(S)
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

def sumsqr(x):
    return np.sum(x**2, axis=0)

def sumsqt(x):
    return np.sum(x**2)

def meansqr(x):
    return np.sum(x**2, axis=0)/len(x)

def meansqt(x):
    return np.sum(x**2)/(x.shape[0] * x.shape[1])

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


def polychor_thresh(X):
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

def polychor_probs(a, b, r):
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

def polychor_loglike(X, a, b, r):
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
    pr = polychor_probs(a, b, r)
    if len(pr.shape)>=3:
        pr = pr[:, :, 0]
    n, k = pr.shape
    pr = np.array([[pr[i, j]+pr[i-1,j-1]-pr[i-1,j]-pr[i,j-1] 
                   for j in range(1,k)] for i in range(1,n)])
    pr = np.maximum(pr, 1e-16)
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
    X = _check_np(X)
    rho = params[0]
    a, b = params[1:k+1], params[k+1:]
    return -polychor_loglike(X, a, b, rho)



def polychor_partial_ll(rho, X, k, params):
    X = _check_np(X)
    a, b = params[:k], params[k:]
    return -polychor_loglike(X, a, b, rho)

  
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
    p_values = _check_1d(p_values)
    idx = np.argsort(p_values)
    correction = _check_1d(_check_np(empirical_cdf(p_values[idx])))
    p_values[idx] /= correction
    
    if is_pd:
        if type(cols) is str:
            cols = list(cols)
        p_values = pd.DataFrame(p_values, columns=cols, index=ix)
    return p_values
    


def msa(R):
    Ri = np.linalg.pinv(R)
    D = np.diag(1.0/np.sqrt(np.diag(Ri)))
    Q = np.linalg.multi_dot([D, Ri, D])
    Ri = Ri - np.diag(np.diag(Ri))
    Q = Q - np.diag(np.diag(Q))
    
    res = np.linalg.norm(Ri) / (np.linalg.norm(Q) + np.linalg.norm(Ri))
    return res

class RobustCorr:

    
    def fun(self, X):
        X, xcols, xix, x_is_pd = check_type(X)
        
        xdev = X - self.est_loc(X)
        psix = self.mfun(xdev)
        S = self.robust_crossprod(psix)
        D = np.diag(1/np.sqrt(np.diag(S)))
        R = D.dot(S).dot(D)
        if (x_is_pd):
            R = pd.DataFrame(R, index=xcols, columns=xcols)
        return R
    
    __call__ = fun

class QuadrantSignedCorr(RobustCorr):
    
    def __init__(self):
        super(QuadrantSignedCorr, self).__init__()
    
    def est_loc(self, X):
        return np.median(X, axis=0)
    
    def mfun(self, X):
        return np.sign(X)
    
    def robust_crossprod(self, X):
        return X.T.dot(X) / valid_overlap(X, X)

class GeneralRobustCorr(RobustCorr):
    
    def __init__(self, alpha=0.1):
        super(GeneralRobustCorr, self).__init__()
        self.a = alpha*100

    def est_loc(self, X):
        return np.median(X, axis=0)
    
    def mfun(self, X):
        return X
    
    def robust_crossprod(self, X):
        xu = sp.stats.scoreatpercentile(X, 100 - self.a, axis=0)
        xl = sp.stats.scoreatpercentile(X, self.a, axis=0)
                
        X[(X>xu)|(X<xl)] = np.nan
        Xm = np.ma.masked_invalid(X)
        XtX = np.asarray(Xm.T.dot(Xm)) / valid_overlap(X, X)
        return XtX
    
def grcorr(X, alpha=0.1):
    grc = GeneralRobustCorr(alpha)
    R = grc(X)
    return R

def qscorr(X):
    qsc = QuadrantSignedCorr()
    R = qsc(X)
    return R


class MultivarAssociation:
    
    def __init__(self, SSE, SSH, SST, p, q, SSE_inv=None, SST_inv=None):
        self.p, self.q, self.s = p, q, np.min(p, q)
        k = p**2*q**2
        if k<=4:
            g = 1
        else:
            g = np.sqrt((k - 4) / (p**2 + p**2 - 5))
        self.g = g
        if SST_inv is None:
            SST_inv = np.linalg.pinv(SST)
        if SSE_inv is None:
            SSE_inv = np.linalg.pinv(SSE)
        
        self.SSE, self.SSH, self.SST = SSE, SSH, SST
        self.SSE_inv, self.SST_inv = SSE_inv, SST_inv
        self.dfe = SSH.shape[0] - p
    
    def hotelling_lawley(self):
        hlt = np.trace(self.SSH.dot(self.SSE_inv))
        u = hlt / self.s
        eta = u / (1.0 + u)
        return hlt, eta
    def pillai_bartlett_trace(self):
        pbt = np.trace(self.SSH.dot(self.SST_inv))
        eta = pbt / self.s
        return pbt, eta
    
    def wilks_likelihood(self):
        wlk = np.linalg.det(self.SSH.dot(self.SST_inv))
        eta = 1 - np.power(wlk, 1.0/self.g)
        return wlk, eta

    def roy_largest_root(self):
        u, V = np.linalg.eigh(self.SSH.dot(self.SST_inv))
        rlr = np.max(u)
        return rlr, rlr
    

     
        
            
        
        
        
    
    
    