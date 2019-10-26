#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 05:06:46 2019

@author: lukepinkel
"""

import patsy
import pandas as pd
import numpy as np
import scipy as sp
import scipy.special
import scipy.stats
import scipy.optimize
from ..utils import linalg_utils, base_utils
  
def trigamma(x):
    return sp.special.polygamma(1, x)      
    
class MinimalNB2:
    
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.ones(1)/2.0
        self.params = np.concatenate([self.beta, self.varp])
    
    def loglike(self, params, X=None):
        if X is None:
            X = self.X
        y = linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1.0 + a * mu
        lg = sp.special.gammaln(y + v) - sp.special.gammaln(v) - sp.special.gammaln(y + 1)
        ln = y * np.log(mu) + y * np.log(a) - (y + v) * np.log(u)
        ll = np.sum(lg + ln)
        return -ll
    
    def gradient(self, params):
        params = linalg_utils._check_1d(params)
        X, y = self.X, linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        return -g
        
    def var_deriv(self, a, mu, y):
        v = 1/a
        u = 1+a*mu
        r = y-mu
        p = 1/u
        vm = v + mu
        vy = v+y
        a2, a3 = a**-2, a**-3
        a4 = (-a2)**2
        dig = sp.special.digamma(v+y) - sp.special.digamma(v)
        z = (dig + np.log(p) - (a * r) / u)
        trg = a4*(trigamma(vy)-trigamma(v) + a - 1/vm + r/(vm**2))
        res = 2*a3*z + trg
        return -res.sum()
    
    def hessian(self, params):
        X, y = self.X, linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]

        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        
        wbb = mu * (1.0 + a * y) / (u**2)
        Hb = -(X.T * wbb).dot(X)
        
        Hab = -X.T.dot((mu * r) / (u**2))
        
        Ha = np.array([self.var_deriv(a, mu, y)]) 
        H = np.block([[Hb, Hab[:, None]], [Hab[:, None].T, Ha]])
        return -H
    
    def fit(self, verbose=0):
        params = self.params
        optimizer = sp.optimize.minimize(self.loglike, params, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':verbose})
        self.optimize = optimizer
        self.params = optimizer.x
        self.LLA = self.loglike(self.params)
        self.vcov = linalg_utils.einv(self.hessian(self.params))
        self.vcov[-1, -1]=np.abs(self.vcov[-1, -1])
        self.params_se = np.sqrt(np.diag(self.vcov))

    def predict(self, X=None, params=None, b=None):
        if X is None:
            X = self.X
        if b is None:
            b = params[:-1]
        mu_hat = np.exp(X.dot(b))
        return mu_hat
    
        
                                

class NegativeBinomial:
    
    def __init__(self, formula, data):
        Y, X = patsy.dmatrices(formula, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = base_utils.check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = base_utils.check_type(Y)
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.ones(1)/2.0
        self.params = np.concatenate([self.beta, self.varp])
    
    def loglike(self, params, X=None):
        if X is None:
            X = self.X
        y = linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1.0 + a * mu
        lg = sp.special.gammaln(y + v) - sp.special.gammaln(v) - sp.special.gammaln(y + 1)
        ln = y * np.log(mu) + y * np.log(a) - (y + v) * np.log(u)
        ll = np.sum(lg + ln)
        return -ll
    
    def gradient(self, params):
        params = linalg_utils._check_1d(params)
        X, y = self.X, linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(np.log(u)+(a*r)/u + sp.special.digamma(v) - sp.special.digamma(y+v))
        ga /= a**2
        g = np.concatenate([gb, np.array([ga])])
        return -g
        
    def var_deriv(self, a, mu, y):
        v, u, r = 1/a, 1+a*mu, y-mu
        p, vm, vy = 1/u, v+mu, v+y
        a2, a3 = a**-2, a**-3
        a4 = (-a2)**2
        dig = sp.special.digamma(v+y) - sp.special.digamma(v)
        z = (dig + np.log(p) - (a * r) / u)
        trg = a4*(trigamma(vy)-trigamma(v) + a - 1/vm + r/(vm**2))
        res = 2*a3*z + trg
        return -res.sum()
    
    def hessian(self, params):
        X, y = self.X, linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]

        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        
        wbb = mu * (1.0 + a * y) / (u**2)
        Hb = -(X.T * wbb).dot(X)
        
        Hab = -X.T.dot((mu * r) / (u**2))
        
        Ha = np.array([self.var_deriv(a, mu, y)]) 
        H = np.block([[Hb, Hab[:, None]], [Hab[:, None].T, Ha]])
        return -H
    
    def deviance(self, params):
        X, y = self.X, linalg_utils._check_1d(self.Y)
        b, a = params[:-1], params[-1]
        mu = np.exp(X.dot(b))
        v = 1.0 / a
        ix = y>0
        dev1 = y[ix] * np.log(y[ix] / mu[ix])
        dev1 -= (y[ix]+v) * np.log((y[ix] + v) / (mu[ix] + v))
        
        dev2 = np.log(1 + a * mu[~ix]) / a
        dev = 2.0 * (np.sum(dev1) + np.sum(dev2))
        return dev
    
    def var_mu(self, params):
        X = self.X
        b, a = params[:-1], params[-1]
        mu = np.exp(X.dot(b))
        v = mu + a * mu**2
        return v
        
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr', 
                                'options':{'verbose':0}}
        intercept_model = MinimalNB2(np.ones((self.n_obs ,1)), self.Y)
        intercept_model.fit()
        self.LL0 = intercept_model.LLA
        params = self.params
        optimizer = sp.optimize.minimize(self.loglike, params, jac=self.gradient, 
                             hess=self.hessian, **optimizer_kwargs)
        
        self.optimizer = optimizer
        self.params = optimizer.x
        self.LLA = self.loglike(self.params)
        self.vcov = linalg_utils.einv(self.hessian(self.params))
        self.vcov[-1, -1]=np.abs(self.vcov[-1, -1])
        self.params_se = np.sqrt(np.diag(self.vcov))
        self.res = pd.DataFrame(np.vstack([self.params, self.params_se]),
                                columns=self.xcols.tolist()+['variance']).T
        self.LLR = -(self.LLA - self.LL0)
        self.BIC = -(np.log(self.n_obs) * len(self.params) - 2 * self.LLA)
        self.AIC = -(2 * len(self.params) - 2 * self.LLA)
        self.dev = self.deviance(self.params)
        self.yhat = self.predict(params=self.params)
        chi2 = (linalg_utils._check_1d(self.Y) - self.yhat)**2
        chi2/= self.var_mu(self.params)
        self.chi2 = np.sum(chi2)
        self.scchi2 = self.chi2 / (self.n_obs - self.n_feats)
        self.chi2_p = sp.stats.chi2.sf(self.chi2,  (self.n_obs - self.n_feats))
        self.dev_p = sp.stats.chi2.sf(self.dev,  (self.n_obs - self.n_feats))
        self.LLRp = sp.stats.chi2.sf(self.LLR,  (self.n_obs - self.n_feats))
        n, p = self.X.shape
        yhat = self.predict(params=self.params)
        self.ssr =np.sum((yhat - yhat.mean())**2)
        rmax =  (1 - np.exp(-2.0/n * (self.LL0)))
        rcs = 1 - np.exp(2.0/n*(self.LLA-self.LL0))
        rna = rcs / rmax
        rmf = 1 - self.LLA/self.LL0
        rma = 1 - (self.LLA-p)/self.LL0
        rmc = self.ssr/(self.ssr+3.29*n)
        rad = self.LLR/(self.LLR+n)
        
        ss = [[self.AIC, self.BIC, self.chi2, self.dev, self.LLR, self.scchi2,
               rcs, rna, rmf, rma, rmc, rad],
              ['-', '-', self.chi2_p, self.dev_p, self.LLRp, '-', '-', '-',
               '-', '-', '-', '-']]
        ss = pd.DataFrame(ss).T
        ss.index = ['AIC', 'BIC', 'chi2', 'deviance', 'LLR', 'scaled_chi2', 
                    'R2_Cox_Snell', 'R2_Nagelkerke', 'R2_McFadden',
                    'R2_McFaddenAdj', 'R2_McKelvey', 'R2_Aldrich']
        ss.columns = ['Test_stat', 'P-value']
        self.sumstats = ss
                     
    def predict(self, X=None, params=None, b=None):
        if X is None:
            X = self.X
        if b is None:
            b = params[:-1]
        mu_hat = np.exp(X.dot(b))
        return mu_hat
    
        
                                
    