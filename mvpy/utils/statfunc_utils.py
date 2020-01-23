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
from .linalg_utils import _check_1d, _check_np, dmat, vechc, normdiff

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

def multivariate_association_tests(rho2, a, b, dfe):
        a2, b2 = a**2, b**2
        if a2*b2 <= 4:
            g = 1
        else:
            g = np.sqrt((a2*b2-4) / (a2 + b2 - 5))
        
        s = np.min([a, b])
        tst_hlt = np.sum(rho2/(1-rho2))
        tst_pbt = np.sum(rho2)
        tst_wlk = np.product(1-rho2)
        tst_rlr = np.max(rho2/(1-rho2))
        
        eta_hlt = (tst_hlt/s) / (1 + tst_hlt/s)
        eta_pbt = tst_pbt / s
        eta_wlk = 1 - np.power(tst_wlk, (1/g))
        eta_rlr = np.max(rho2)
        
        test_stats = np.vstack([tst_hlt, tst_pbt, tst_wlk,]).T
        effect_sizes = np.vstack([eta_hlt, eta_pbt, eta_wlk]).T
        test_stats = pd.DataFrame(test_stats, columns=['HLT', 'PBT', 'WLK'])
        effect_sizes = pd.DataFrame(effect_sizes, columns=['HLT', 'PBT', 'WLK'])
        
        df_hlt1 = a * b
        df_wlk1 = a * b
        df_rlr1 = np.max([a, b])

        df_pbt1 = s * (dfe + s - b) * (dfe + a + 2) * (dfe + a - 1)
        df_pbt1 /= (dfe * (dfe + a - b))
        df_pbt1 -= 2
        df_pbt1 *= (a * b) / (s * (dfe + a))

        df_hlt2 = (dfe**2 - dfe * (2 * b + 3) + b * (b + 3)) * (a * b + 2)
        df_hlt2 /= (dfe * (a + b + 1) - (a + 2 * b + b2 - 1))
        df_hlt2 += 4

        df_pbt2 = s * (dfe + s - b) * (dfe + a + 2) * (dfe + a - 1)
        df_pbt2 /= dfe * (dfe + a - b)
        df_pbt2 -= 2
        df_pbt2 *= (dfe + s - b) / (dfe + a)

        df_wlk2 = g * (dfe - (b - a + 1) / 2) - (a * b - 2) / 2
        df_rlr2 = dfe - np.max([a, b]) + b
        df1 = np.array([df_hlt1, df_pbt1, df_wlk1])
        df2 = np.array([df_hlt2, df_pbt2, df_wlk2])
        f_values = (effect_sizes / df1) / ((1 - effect_sizes) / df2)
        p_values = sp.stats.f.sf(f_values, df1, df2)
        p_values = pd.DataFrame(p_values, columns=effect_sizes.columns)
        df1 = pd.DataFrame(df1, index=effect_sizes.columns).T
        df2 = pd.DataFrame(df2, index=effect_sizes.columns).T
        f_rlr = tst_rlr * (df_rlr2 / df_rlr1)
        p_rlr = sp.stats.f.sf(f_rlr, df_rlr1, df_rlr2)
        rlr = pd.DataFrame([tst_rlr, eta_rlr, f_rlr, df_rlr1, df_rlr2, p_rlr])
        rlr.index = ['Test Stat', 'Eta', 'F-values', 'df1', 'df2', 'P-values']
        rlr.columns = ['RLR']
        rlr = rlr.T
        sumstats = pd.concat([test_stats, effect_sizes, f_values, df1, df2, 
                              p_values])
        sumstats.index = ['Test Stat', 'Eta', 'F-values', 'df1', 'df2', 'P-values']
        sumstats = sumstats.T
        sumstats = pd.concat([sumstats, rlr])
        return sumstats, effect_sizes, rho2



def multivariate_kurtosis(X):
    m = np.mean(X, axis=0)
    N = X.shape[0]
    n = N - 1.0
    Z = X - m
    S = Z.T.dot(Z) / n
    W = np.linalg.pinv(S)
    eta = 0.0
    p = (X.shape[1]*(X.shape[1]+2))
    for i in range(X.shape[0]):
        Zi = Z[i]
        eta += (Zi.T.dot(W).dot(Zi)**2)/p
    eta/=X.shape[0]
    return eta

def multivar_marginal_kurtosis(X):
    k = sp.stats.moment(X, 4)/(sp.stats.moment(X, 2)**2)/3.0
    return k



def cov_sample_cov(X=None, S=None, excess_kurt=None, kurt=None):
    if S is None:
        m = np.mean(X, axis=0)
        N = X.shape[0]
        n = N - 1.0
        Z = X - m
        S = Z.T.dot(Z) / n
    if kurt is None:
        if excess_kurt is None:
            if X is None:
                kurt = np.zeros(X.shape[1])
            else:
                kurt = multivar_marginal_kurtosis(X)
        else:
            kurt = (excess_kurt + 3.0) / 3.0
    D = np.linalg.pinv(dmat(S.shape[0]))
    u = np.atleast_2d(np.sqrt(kurt))
    A = 0.5 * (u + u.T)
    C = A*S
    v = np.concatenate([vechc(C), vechc(S)], axis=1)
    M = np.eye(2)
    M[1, 1] = -1
    V = 2*D.dot(np.kron(C, C)).dot(D.T)+v.dot(M).dot(v.T)
    return V      
    
def trim_extremes(x, alpha=10):
    if alpha>1:
        alpha/=100
    n = x.shape[0]
    k = int(np.round(alpha * n))
    order = x.argsort()
    x = x[order]
    x = x[k:-k]
    return x

def trimmed_mean(x, alpha=10):
    x = trim_extremes(x, alpha=alpha)
    return np.mean(x)


def MedScale(r):
    s = np.median(np.abs(r - np.median(r))) / sp.stats.norm.ppf(.75)
    return s
    

class Huber:
    
    def __init__(self, scale_estimator=MedScale):
        self.c0 = 1.345
        self.c1 = 0.6745
        self._scale_estimator = MedScale
        
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixa] = u[ixa]**2
        v[ixb] = np.abs(2*u[ixb])*self.c0 - self.c0**2
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixb] = self.c0 * np.sign(u[ixb])
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = u.copy()
        ixa = np.abs(u) <= self.c0
        ixb = ~ixa
        v[ixa] = 1
        v[ixb] = 0
        return v
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = u.copy()
        ixa = np.abs(u) < self.c0
        ixb = ~ixa
        v[ixa] = 1
        v[ixb] = self.c0 / np.abs(u[ixb])
        return v
    
    def estimate_scale(self, r):
        return self._scale_estimator(r)
        
class Bisquare:
    
    def __init__(self, scale_estimator=MedScale):
        self.c0 = 4.685
        self.c1 = 0.6745
        self._scale_estimator = scale_estimator
    
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) < c
        ixb = ~ixa
        v[ixa] = c**2 / 3 * (1 - ((1 - (u[ixa] / c)**2)**3))
        v[ixb] = 2 * c
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) <= c
        ixb = ~ixa
        v[ixa] = u[ixa] * (1 - (u[ixa] / c)**2)**2
        v[ixb] = 0
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) <= self.c0
        ixb = ~ixa
        u2c2 = (u**2 / c**2)
        v[ixa] = (1 -u2c2[ixa]) * (1 - 5 * u2c2[ixa])
        v[ixb] = 0
        return v
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = u.copy()
        c = self.c0
        ixa = np.abs(u) < c
        ixb = ~ixa
        v[ixa] = (1 - (u[ixa] / c)**2)**2
        v[ixb] = 0
        return v
     
    def estimate_scale(self, r):
        return self._scale_estimator(r)
        
class Hampel:
    
    def __init__(self, k=0.9016085, scale_estimator=MedScale):
        self.a = 1.5 * k
        self.b = 3.5 * k
        self.r = 8.0 * k
        self.k = k
        self.c = self.a / 2.0 * (self.b - self.a + self.r)
        self.a2 = self.a**2
        self._scale_estimator = scale_estimator
    
    def rho_func(self, u):
        '''
        Function to be minimized
        '''
        a, a2, b, c, r = self.a, self.a2, self.b, self.c, self.r
        v = u.copy()
        au = np.abs(u)
        ixa = au <= a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        ixd = au>r
        v[ixa] = 0.5 * u[ixa]**2 / c
        v[ixb] = (0.5 * a2 + a*(au[ixb] - a)) / c
        v[ixc] = 0.5 * (2*b-a+(au[ixc]-b)*(1+(r-au[ixc])/(r-b))) / c
        v[ixd] = 1.0
        return v
    
    def psi_func(self, u):
        '''
        Derivative of rho
        '''
        v = u.copy()
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        sgnu = np.sign(u)
        ixa = au <= self.a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        ixd = au>r
        v[ixa] = u[ixa]
        v[ixb] = a * sgnu[ixb]
        v[ixc] = a * sgnu[ixc] * (r - au[ixc]) / (r - b)
        v[ixd] = 0
        return v
    
    def phi_func(self, u):
        '''
        Second derivative of rho
        '''
        v = np.zeros(u.shape[0])
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        ixa = au <= self.a
        ixc = (au>b) * (au<=r)
        v[ixa] = 1.0
        v[ixc] = (a * np.sign(u)[ixc] * u[ixc]) / (au[ixc] * (r - b))
        return v
    
    def weights(self, u):
        '''
        Equivelant to psi(u)/u
        '''
        v = np.zeros(u.shape[0])
        a, b, r = self.a, self.b, self.r
        au = np.abs(u)
        ixa = au <= self.a
        ixb = (au>a) * (au<=b)
        ixc = (au>b) * (au<=r)
        v[ixa] = 1.0
        v[ixb] = a / au[ixb]
        v[ixc] = a * (r - au[ixc]) / (au[ixc] * (r - b))
        return v
      
    def estimate_scale(self, r):
        return self._scale_estimator(r)
    




class Laplace:
    
    def __init__(self, scale_estimator=MedScale):
        self.a = 1.0
        self._scale_estimator = scale_estimator
     
    def rho_func(self, u):
        rho = np.abs(u)
        return rho

    def psi_func(self, u):
        psi = np.sign(u)
        return psi

    def phi_func(self, u):
        phi = np.ones_like(u)
        return phi

    def weights(self, u):
        w = self.psi_func(u) / u
        return w
       
    def estimate_scale(self, r):
        return self._scale_estimator(r)   
    
    
    
class Lpnorm:
    
    def __init__(self, p=1.5, scale_estimator=MedScale):
        self.p = p
        self.a = p - 1.0
        self.b = p / 2.0
        self.c = self.a * self.b
        self.d = p - 2.0
        self._scale_estimator = scale_estimator
     
    def rho_func(self, u):
        rho = 0.5 * np.abs(u)**self.p
        return rho

    def psi_func(self, u):
        psi = self.b * np.abs(u)**self.a
        psi*= np.sign(u)
        return psi

    def phi_func(self, u):
        phi = -np.abs(u)**self.d * self.c * np.sign(u)
        return phi

    def weights(self, u):
        w = self.psi_func(u) / u
        return w
          
    def estimate_scale(self, r):
        return self._scale_estimator(r)
    
    
    
class Cauchy:
    
    def __init__(self, p=1.5, scale_estimator=MedScale):
        self.p = p
        self.a = p - 1.0
        self.b = p / 2.0
        self.c = self.a * self.b
        self.d = p - 2.0
        self._scale_estimator = scale_estimator
     
    def rho_func(self, u):
        rho = np.log(1 + u**2)
        return rho

    def psi_func(self, u):
        psi = 2 * u / (1 + u**2)
        return psi

    def phi_func(self, u):
        u2 = u**2
        phi = 2 * (u2 - 1) / (u2 + 1)**2
        return phi

    def weights(self, u):
        w = self.psi_func(u) / u
        return w
          
    def estimate_scale(self, r):
        return self._scale_estimator(r)






def _m_est_loc(Y, n_iters=100, tol=1e-9, method=Huber()):
    X = np.ones((Y.shape[0], 1))
    w = np.ones((Y.shape[0], 1))
    b0 = np.zeros(1)
    for i in range(n_iters):
        if w.ndim==1:
            w = w.reshape(w.shape[0], 1)
        Xw = X * w
        XtWX_inv = np.linalg.pinv(np.dot(Xw.T, X))
        beta = XtWX_inv.dot(np.dot(Xw.T, Y))
        r = _check_1d(Y) - _check_1d(X.dot(beta))
        s = np.median(np.abs(r - np.median(r))) / sp.stats.norm.ppf(.75)
        u = r / s
        w = method.weights(u)
        
        db = normdiff(beta, b0)
        if db < tol:
            break
        b0 = beta
    return beta







