#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 05:06:46 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
from numpy import log, exp, dot, diag
from numpy.linalg import pinv, inv
from scipy.special import gammaln, digamma, polygamma, gamma
from mvpy.utils.linalg_utils import _check_1d, _check_2d, einv
from mvpy.utils.base_utils import check_type
from patsy import dmatrices
import statsmodels.api as sm
from scipy.optimize import minimize 
np.set_printoptions(suppress=True)

    
  
def trigamma(x):
    return polygamma(1, x)      
    

class NegativeBinomial:
    
    def __init__(self, formula, data):
        Y, X = dmatrices(formula, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = X.shape
        self.beta = np.zeros(self.n_feats)
        self.varp = np.ones(1)/2.0
        self.params = np.concatenate([self.beta, self.varp])
    
    def loglike(self, params):
        X, y = self.X, _check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1.0 + a * mu
        lg = gammaln(y + v) - gammaln(v) - gammaln(y + 1)
        ln = y * log(mu) + y * log(a) - (y + v) * log(u)
        ll = np.sum(lg + ln)
        return -ll
    
    def gradient(self, params):
        params = _check_1d(params)
        X, y = self.X, _check_1d(self.Y)
        b, a = params[:-1], params[-1]
        v = 1.0 / a
        mu = np.exp(X.dot(b))
        u = 1 + a * mu
        r = y-mu
        gb = X.T.dot(r / u)
        ga = np.sum(log(u)+(a*r)/u + digamma(v) - digamma(y+v))
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
        dig = digamma(v+y) - digamma(v)
        z = (dig + log(p) - (a * r) / u)
        trg = a4*(trigamma(vy)-trigamma(v) + a - 1/vm + r/(vm**2))
        res = 2*a3*z + trg
        return -res.sum()
    
    def hessian(self, params):
        X, y = self.X, _check_1d(self.Y)
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
    
    def fit(self, verbose=2):
        params = self.params
        optimizer = minimize(self.loglike, params, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':verbose})
        self.optimize = optimizer
        self.params = optimizer.x
        self.vcov = einv(self.hessian(self.params))
        self.vcov[-1, -1]=np.abs(self.vcov[-1, -1])
        self.params_se = np.sqrt(np.diag(self.vcov))
        self.res = pd.DataFrame(np.vstack([self.params, self.params_se]),
                                columns=self.xcols.tolist()+['variance']).T
                                
    