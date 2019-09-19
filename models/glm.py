#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:25:42 2019

@author: lukepinkel
"""
import pandas as pd
import numpy as np
from numpy import log, exp
from patsy import dmatrices


from scipy.optimize import minimize
from scipy.stats import t as t_dist, chi2 as chi2_dist

from ..utils.base_utils import check_type
from ...utils.linalg_utils import einv, _check_1d, _check_0d


class minimal_logistic:
    
    def __init__(self, X, Y):
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.dfe = self.X.shape[0] - self.X.shape[1]
    def mean_func(self, X):
        P = exp(X)
        return P/(1.0+P)
    
    def mean_func_prime(self, X):
        P = exp(X)
        return P/((1.0+P)**2.0)
    
    def loglike(self, beta):
        X, Y = self.X, self.Y
        eta = self.mean_func(X.dot(beta))
        a1, a2 = log(np.maximum(eta, 1e-16)), log(np.maximum(1.0-eta, 1e-16))
        LL = Y.T.dot(a1) + (1.0 - Y).T.dot(a2)
        return -_check_0d(LL)
    
    def gradient(self, beta):
        X, Y = self.X, self.Y
        eta = self.mean_func(X.dot(beta))
        return -_check_1d(X.T.dot(_check_1d(Y)-eta))
    
    def hessian(self, beta):
        X = self.X
        W = np.diag(self.mean_func_prime(X.dot(beta)))
        H = X.T.dot(W).dot(X)
        return H

    def fit(self):
        theta = np.ones(self.X.shape[1])
        optimizer = minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':0})
        self.optimizer = optimizer
        self.beta = optimizer.x
    
    

class Logistic:
    
    def __init__(self, frm=None, data=None, X=None, Y=None):
        if frm is not None:
            Y, X = dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.dfe = self.X.shape[0] - self.X.shape[1]
    def mean_func(self, X):
        P = exp(X)
        return P/(1.0+P)
    
    def mean_func_prime(self, X):
        P = exp(X)
        return P/((1.0+P)**2.0)
    
    def loglike(self, beta):
        X, Y = self.X, self.Y
        eta = self.mean_func(X.dot(beta))
        a1, a2 = log(np.maximum(eta, 1e-16)), log(np.maximum(1.0-eta, 1e-16))
        LL = Y.T.dot(a1) + (1.0 - Y).T.dot(a2)
        return -_check_0d(LL)
    
    def gradient(self, beta):
        X, Y = self.X, self.Y
        eta = self.mean_func(X.dot(beta))
        return -_check_1d(X.T.dot(_check_1d(Y)-eta))
    
    def hessian(self, beta):
        X = self.X
        W = np.diag(self.mean_func_prime(X.dot(beta)))
        H = X.T.dot(W).dot(X)
        return H
    
    def fit(self, verbose=2):
        X0 = np.ones((self.X.shape[0], 1))
        intercept_model = minimal_logistic(X0, self.Y)
        intercept_model.fit()
        self.intercept_model = intercept_model
        
        self.LL0 = intercept_model.loglike(intercept_model.beta)
        theta = np.ones(self.X.shape[1])
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
        count = self.Y.T.dot(yhat>0.5)
        psuedo_r2 = {}
        psuedo_r2['Efron'] = 1 - self.sse / self.sst
        psuedo_r2['McFaddens'] = 1 - self.LLA/self.LL0
        psuedo_r2['McFaddens_adj'] = 1 - (self.LLA-p)/self.sst
        psuedo_r2['McKelvey'] = self.ssr/(self.ssr+n)
        psuedo_r2['Aldrich'] = self.LLR/(self.LLR+n)
        psuedo_r2['Count'] = count/np.sum(self.Y)
        self.psuedo_r2 = psuedo_r2
        self.LLRp =  chi2_dist.sf(self.LLR, len(self.beta))
        
        
        
    def predict(self, X=None):
        if X is None:
            X = self.X
        eta = self.mean_func(X.dot(self.beta))
        return eta
    
    
    