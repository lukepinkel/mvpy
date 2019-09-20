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

from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import t as t_dist, chi2 as chi2_dist
from scipy.special import gammaln, digamma, polygamma
from ..utils.base_utils import check_type
from ...utils.linalg_utils import einv, _check_1d, _check_0d

'''
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
    
'''


class MinimalGLM:
    
    def __init__(self, X, Y, family):
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.family = family
        self.YtX = self.Y.T.dot(self.X)
        
    def loglike(self, beta):
        X, YtX, jn = self.X, self.YtX, self.jn
        bt = self.family.cumulant(X.dot(beta))
        LL = YtX.dot(beta) - jn.T.dot(bt)
        return -LL
    
    def gradient(self, beta):
        X, Y =  self.X, _check_1d(self.Y)
        mu = _check_1d(self.family.cumulant_prime(X.dot(beta)))
        g = -(Y - mu).T.dot(X)
        return g
    
    def hessian(self, beta):
        X = self.X
        W = np.diag(self.family.cumulant_double_prime(X.dot(beta)))
        H = X.T.dot(W).dot(X)
        return H
    
    def fit(self, verbose=2):
        theta = np.ones(self.X.shape[1])/2.0
        optimizer = minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr',
                             options={'verbose':0})
        self.optimizer = optimizer
        self.beta = optimizer.x
        
        
        
        
        


class GLM:
    
    def __init__(self, frm=None, data=None, X=None, Y=None, family=None):
        '''
        examples:
            
        data = pd.read_csv("/users/lukepinkel/Downloads/poisson_sim.csv")
        frm="num_awards ~ C(prog) + math"
        model = GLM(frm=frm, data=data, family='poisson')
        model.fit()
        
        import statsmodels.api as sm
        spector_data = sm.datasets.spector.load_pandas()
        spector_data.exog = sm.add_constant(spector_data.exog)
        data = pd.concat([spector_data.endog, spector_data.exog], axis=1)
        frm = "GRADE ~ GPA+TUCE+PSI"
        model = GLM(frm=frm, data=data, family='binomial')
        model.fit()
        
        
        
        '''
        if family.lower() in ['binomial', 'binom', 'logit', 'logistic', 'binary', 
                      'bernoulli', 'bern']:
            self.family = LogisticGLM()
        elif family.lower() in ['poisson', 'pois', 'count']:
            self.family = PoissonGLM()   
        if frm is not None:
            Y, X = dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        self.theta_init = np.ones(self.n_feats)/self.n_feats
        
        
    def loglike(self, beta):
        X, YtX, jn = self.X, self.YtX, self.jn
        bt = self.family.cumulant(X.dot(beta))
        LL = YtX.dot(beta) - jn.T.dot(bt)
        return -LL
    
    def gradient(self, beta):
        X, Y =  self.X, _check_1d(self.Y)
        mu = _check_1d(self.family.cumulant_prime(X.dot(beta)))
        g = -(Y - mu).T.dot(X)
        return g
    
    def hessian(self, beta):
        X = self.X
        W = np.diag(self.family.cumulant_double_prime(X.dot(beta)))
        H = X.T.dot(W).dot(X)
        return H
    
    def fit(self):
        X0 = np.ones((self.n_obs, 1))
        intercept_model = MinimalGLM(X0, self.Y, self.family)
        intercept_model.fit()
        self.intercept_model = intercept_model
        self.LL0 = intercept_model.loglike(intercept_model.beta)
        theta = self.theta_init
        optimizer = minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, method='trust-constr')        
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
        psuedo_r2 = {}
        psuedo_r2['Efron'] = 1 - self.sse / self.sst
        psuedo_r2['McFaddens'] = 1 - self.LLA/self.LL0
        psuedo_r2['McFaddens_adj'] = 1 - (self.LLA-p)/self.sst
        psuedo_r2['McKelvey'] = self.ssr/(self.ssr+n)
        psuedo_r2['Aldrich'] = self.LLR/(self.LLR+n)
        self.psuedo_r2 = psuedo_r2
        self.LLRp =  chi2_dist.sf(self.LLR, len(self.beta))
    
    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.family.cumulant_prime(X.dot(self.beta))

        


      
class LogisticGLM:
    
    def __init__(self):
        self.dist = 'b'
    
    def cumulant(self, Xb):
        u = 1+np.exp(Xb)
        bt = np.log(np.maximum(u, 1e-16))
        return bt
    
    def cumulant_prime(self, Xb):
        u = np.exp(Xb)
        db = u / (1 + u)
        return db
    
    def cumulant_double_prime(self, Xb):
        u = np.exp(Xb)
        d2b = u / ((1 + u)**2)
        return d2b


class PoissonGLM:
    
    def __init__(self):
        self.dist = 'p'
    
    def cumulant(self, Xb):
        bt = np.exp(Xb)
        return bt
    
    def cumulant_prime(self, Xb):
        db = np.exp(Xb)
        return db
    
    def cumulant_double_prime(self, Xb):
        d2b = np.exp(Xb)
        return d2b


 '''   
class NegBinomial:
    
    def __init__(self, frm=None, data=None, X=None, Y=None, type_=2):
        '''
        data = pd.read_csv("/users/lukepinkel/Downloads/poisson_sim.csv")
        frm="num_awards ~ C(prog) + math"
        model = NegBinomial(frm=frm, data=data)
        model.loglike(model.theta_init)
        model.gradient(model.theta_init)
        approx_fprime(model.theta_init, model.loglike, 1e-8)
        res = minimize(model.loglike, model.theta_init, 
                       jac=model.gradient, hess=model.hessian,
                       method='trust-constr', options={'maxiter':5000})
        
        '''
        
        if frm is not None:
            Y, X = dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        poisson_model = GLM(frm, data, family='poisson')
        poisson_model.fit()
        theta_init = poisson_model.beta
        theta_init = np.concatenate([theta_init, np.ones(1)])/2
        self.theta_init = theta_init
        self.nbp = 2-type_
        
    def loglike(self, theta):
        X, Y, nbp = self.X, _check_1d(self.Y), self.nbp
        kappa, beta = theta[-1], theta[:-1]
        lmb = np.exp(X.dot(beta))
        Z = kappa * lmb**nbp
        M = kappa / (kappa + lmb)
        L1 = gammaln(Y+Z) - gammaln(Z) - gammaln(1+Y)
        L2 = Z.T.dot(log(M)) + Y.T.dot(log(1-M))
        LL = np.sum(L1)+L2
        return -LL
    
    def gradient(self, theta):
        theta = _check_1d(theta)
        X, Y, nbp = self.X, _check_1d(self.Y), self.nbp
        kappa, beta = theta[-1], theta[:-1]
        lmb = np.exp(X.dot(beta))
        Z = kappa * lmb**nbp
        W = Z / (Z + lmb)
        gb = (kappa * ((Y - lmb) / (kappa + lmb))).T.dot(X)
        dgdk = Z / kappa
        dwdk = (1 / kappa) * W * (1 - W)
        dk1 = (digamma(Y+Z) - digamma(Z) + np.log(W)).T.dot(dgdk)
        dk2 = (Z / W - Y / (1 - W)).T.dot(dwdk)
        gk = dk1+dk2
        g = np.concatenate([gb, -np.array([gk])])
        return -g
    
    def hessian(self, theta):
        X, Y, n = self.X, _check_1d(self.Y), self.n_obs
        kappa, beta = theta[-1], theta[:-1]
        lmb = np.exp(X.dot(beta))
        u = lmb * (kappa+Y)
        v = (kappa + lmb)**2
        Hb = kappa * X.T.dot(np.diag(-u/v)).dot(X)
        Hbk = ((Y - lmb) * (kappa / (kappa + lmb))).T.dot(X)
        Hk1 = polygamma(1, Y+kappa) - (1 / (lmb+kappa))
        Hk2 = n*(1/kappa+polygamma(1, kappa))
        Hk = np.array([[Hk2]])+np.sum(Hk1)
        H = np.block([[Hb, Hbk[:, None]], [Hbk[:, None].T, Hk]])
        return -H
        
    
    
    
        
   '''     
        











   
    