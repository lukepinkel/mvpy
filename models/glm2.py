#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:08:10 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
from numpy import log, exp
from patsy import dmatrices
from collections import OrderedDict
from numpy.linalg import pinv
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import t as t_dist, chi2 as chi2_dist
from scipy.special import gammaln, digamma, polygamma
from ..utils.base_utils import check_type
from ..utils.linalg_utils import (einv, _check_1d, _check_0d, _check_np,
                                   _check_2d)

import scipy.stats as spstats


class MinimalGLM:
    
    def __init__(self, X, Y, fam):
        self.f = fam
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        
    def loglike(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y, jn = self.X, self.Y, self.jn
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        Z = self.f.cumulant(T)
        LL = (Y.T.dot(T) - jn.T.dot(Z)) / phi
        return -_check_0d(LL)
    
    def gradient(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv =1.0/V
        W = Vinv * self.f.dinv_link(eta)
        G = -(_check_1d(Y) - mu) * W
        g = X.T.dot(G)
        return g
    
    def hessian(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv = 1.0/V
        W0 = self.f.dinv_link(eta)**2
        W1 = self.f.d2inv_link(eta)
        W2 = self.f.d2canonical(mu)
        
        Psc = (_check_1d(Y)-mu) * (W2*W0+W1*Vinv)
        Psb = Vinv*W0
        W = Psc - Psb
        
        H = (X.T * W).dot(X)
        return -H
    
    def fit(self, verbose=2):
        theta = np.ones(self.X.shape[1])/2.0
        optimizer = minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':0})
        self.optimizer = optimizer
        self.beta = optimizer.x
        
        
class GLM:
    
    def __init__(self, frm=None, data=None, fam=None):
        '''
        Generalized linear model class.  Currently supports
        dependent Bernoulli and Poisson variables, and 
        logit, probit, log, and reciprocal link functions.
        
        Inverse Gaussian, and Gamma will be added to distributions
        Cloglog, and identity link functions will be added
        Parameters
        -----------
            frm: string, formula
            data: dataframe
            fam: class of the distribution being modeled  
        
        '''
        self.f = fam
        Y, X = dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        self.theta_init = np.ones(self.n_feats)/self.n_feats
        
        
    def loglike(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y, jn = self.X, self.Y, self.jn
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        Z = self.f.cumulant(T)
        LL = (Y.T.dot(T) - jn.T.dot(Z)) / phi
        return -_check_0d(LL)
    
    def gradient(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv =1.0/V
        W = Vinv * self.f.dinv_link(eta)
        G = -(_check_1d(Y) - mu) * W
        g = X.T.dot(G)
        return g
    
    def hessian(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv = 1.0/V
        W0 = self.f.dinv_link(eta)**2
        W1 = self.f.d2inv_link(eta)
        W2 = self.f.d2canonical(mu)
        
        
        Psc = (_check_1d(Y)-mu) * (W2*W0+W1*Vinv)
        Psb = Vinv*W0
        W = Psc - Psb
        
        H = (X.T * W).dot(X)
        return -H
    
    def fit(self, verbose=0):
        X0 = np.ones((self.n_obs, 1))
        intercept_model = MinimalGLM(X0, self.Y, self.f)
        intercept_model.fit()
        self.intercept_model = intercept_model
        self.LL0 = intercept_model.loglike(intercept_model.beta)
        theta = self.theta_init
        optimizer = minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':verbose})        
        self.optimizer = optimizer
        self.beta = optimizer.x
        self.hess = self.hessian(self.beta)
        self.grad = self.gradient(self.beta)
        self.vcov = einv(self.hess)
        self.beta_se = np.sqrt(np.diag(self.vcov))
        self.tvals = self.beta/self.beta_se
        self.pvals = t_dist.sf(abs(self.tvals), self.dfe)*2.0
        self.res = pd.DataFrame(np.vstack([self.beta, self.beta_se,
                                           self.tvals, self.pvals]).T, 
                                           index=self.xcols, 
                                           columns=['beta', 'SE', 't', 'p'])
        self.LLA = self.loglike(self.beta)
        self.LLR = 2.0*(self.LL0-self.LLA)
        self.sse = np.sum((self.Y[:, 0]-self.predict())**2)
        self.sst = np.var(self.Y)*self.Y.shape[0]
        n, p = self.X.shape[0], self.X.shape[1]
        yhat = self.predict()
        self.ssr =np.sum(yhat**2)
        pseudo_r2 = {}
        pseudo_r2['Efron'] = 1 - self.sse / self.sst
        pseudo_r2['McFaddens'] = 1 - self.LLA/self.LL0
        pseudo_r2['McFaddens_adj'] = 1 - (self.LLA-p)/self.sst
        pseudo_r2['McKelvey'] = self.ssr/(self.ssr+n)
        pseudo_r2['Aldrich'] = self.LLR/(self.LLR+n)
        self.pseudo_r2 = pseudo_r2
        self.LLRp =  chi2_dist.sf(self.LLR, len(self.beta))
    
    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.f.inv_link(X.dot(self.beta))
    
    
class Bernoulli:
    
    def __init__(self, link='canonical'):
        if link is 'canonical':
            self.link=LogitLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
    def canonical_parameter(self, mu):
        u = mu / (1  - mu)
        T = np.log(u)
        return T
    
    def inv_link(self, eta):
        return self.link.inv_link(eta)
        
    def dinv_link(self, eta):
        return self.link.dinv_link(eta)
    
    def d2inv_link(self, eta):
        return self.link.d2inv_link(eta)
    
    def cumulant(self, T):
        u = 1 + np.exp(T)
        b = np.log(u)
        return b
    
    def mean_func(self, T):
        u = np.exp(T)
        mu = u / (1 + u)
        return mu
    
    def var_func(self, T):
        mu = self.mean_func(T)
        V = _check_1d(mu * (1 - mu))
        return V
                
    def d2canonical(self, mu):
        res = 1.0/((1.0 - mu)**2)-1.0/(mu**2)
        return res
    
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    

class Poisson:
    
    def __init__(self, link='canonical'):
        if link is 'canonical':
            self.link=LogLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
    def canonical_parameter(self, mu):
        T = np.log(mu)
        return T
    
    def inv_link(self, eta):
        return self.link.inv_link(eta)
        
    def dinv_link(self, eta):
        return self.link.dinv_link(eta)
    
    def d2inv_link(self, eta):
        return self.link.d2inv_link(eta)
    
    def cumulant(self, T):
        b = np.exp(T)
        return b
    
    def mean_func(self, T):
        mu = np.exp(T)
        return mu
    
    def var_func(self, T):
        mu = self.mean_func(T)
        V = _check_1d(mu)
        return V
                
    def d2canonical(self, mu):
        res = -1  /(mu**2)
        return res
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    
class LogitLink:

    def __init__(self):
        self.fnc='logit'
        
    def inv_link(self, eta):
        u = np.exp(eta)
        mu = u / (u + 1)
        return mu
    
    def dinv_link(self, eta):
        u = np.exp(eta)
        dmu = u / ((1 + u)**2)
        return dmu
        
    def d2inv_link(self, eta):
        u = np.exp(eta)
        d2mu = -(u * (u - 1.0)) / ((1.0 + u)**3)
        return d2mu

class ProbitLink:
    
    def __init__(self):
        self.fnc='probit'
        
    def inv_link(self, eta):
        mu = spstats.norm.cdf(eta, loc=0, scale=1)
        mu[mu==1.0]-=1e-16
        return mu
    
    def dinv_link(self, eta):
        dmu = spstats.norm.pdf(eta, loc=0, scale=1)
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = -eta * spstats.norm.pdf(eta, loc=0, scale=1)
        return d2mu

class LogLink:
    def __init__(self):
        self.fnc='log'
        
    def inv_link(self, eta):
        mu = np.exp(eta)
        return mu
    
    def dinv_link(self, eta):
        dmu = np.exp(eta)
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = np.exp(eta)
        return d2mu
    
class ReciprocalLink:
    
    
    def __init__(self):
        self.fnc='reciprocal'
    
    def inv_link(self, eta):
        mu = 1 / (eta)
        return mu
    
    def dinv_link(self, eta):
        dmu = -1 / (eta**2)
        return dmu
    def d2inv_link(self, eta):
        d2mu = 2 / (eta**3)
        return d2mu

class CloglogLink:
    
    def __init__(self):
        self.fnc='cloglog'
        
    def inv_link(self, eta):
        mu = 1.0-np.exp(-np.exp(eta))
        return mu
    
    def dinv_link(self, eta):
        dmu = np.exp(eta-np.exp(eta))
        return dmu
    def d2inv_link(self, eta):
        d2mu = -np.exp(eta - np.exp(eta)) * (np.exp(eta) - 1.0)
        return d2mu

class PowerLink:
    
    def __init__(self, alpha):
        self.fnc = 'power'
        self.alpha=alpha
        
    def inv_link(self, eta):
        if self.alpha==0:
            mu = np.exp(eta)
        else:
            mu = eta**(1/self.alpha)
        return mu
    
    def dinv_link(self, eta):
        if self.alpha==0:
            dmu = np.exp(eta)
        else:
            dmu = -(eta**(1/self.alpha) * np.log(eta)) / (self.alpha**2)
        return dmu
    
    def d2inv_link(self, eta):
        alpha=self.alpha
        lnx = np.log(eta)
        if alpha==0:
            d2mu = np.exp(eta)
        else:
            d2mu = (eta**(1/alpha) * lnx * (lnx+2*alpha)) / alpha**4
        return d2mu
    
    

    
    
    

    

    
    
    
