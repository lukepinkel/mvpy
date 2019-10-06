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
from collections import OrderedDict
from numpy.linalg import pinv
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import t as t_dist, chi2 as chi2_dist
from scipy.special import gammaln, digamma, polygamma
from ..utils.base_utils import check_type
from ...utils.linalg_utils import (einv, _check_1d, _check_0d, _check_np,
                                   _check_2d)


class LM:
    
    def __init__(self, formula, data):
        y, X = dmatrices(formula, data=data, return_type='dataframe')
        self.X, self.y = X, y
        self.sumstats = self.lmss(X, y)
        self.gram = einv(X.T.dot(X))
        self.coefs = _check_np(self.gram.dot(X.T).dot(_check_np(y)))
        self.yhat = _check_np(X).dot(self.coefs)
        self.error_var = np.sum((_check_np(y) - self.yhat)**2, 
                                axis=0)/(X.shape[0]-X.shape[1]-1)
        self.coefs_se = np.sqrt(np.diag(self.gram*self.error_var))
        self.ll = self.loglike(self.coefs, self.error_var)
        beta = np.concatenate([_check_2d(self.coefs), _check_2d(self.coefs_se)],
                               axis=1)
        self.res = pd.DataFrame(beta, index=self.X.columns, columns=['beta', 'SE'])
        self.res['t'] = self.res['beta'] / self.res['SE']
        self.res['p'] = t_dist.sf(abs(self.res['t']), X.shape[0]-X.shape[1])*2.0
        
    
    def lmss(self, X, y):
        di = X.design_info
        #if 'Intercept' in X.columns.tolist():
        #    dnames = di.term_names[1:]
        #else:
        #    dnames = di.term_names
        Xmats = [X.loc[:, di.subset(x).column_names] for x in di.term_names[1:]]
        #Xmats = [X.iloc[:, di.slice(x)] for x in dnames]
        Xmats = OrderedDict(zip(di.term_names[1:], Xmats))
        
        anv = OrderedDict()
        for key in Xmats.keys():
            anv[key] = np.concatenate(self.minimum_ols(Xmats[key], y))
        
        anova_table = pd.DataFrame(anv, index=['sst', 'ssr', 'sse', 'mst', 'msr',
                                               'mse', 'r2', 'r2_adj']).T
            
        anova_table['F'] = anova_table.eval('msr/mse')
        return anova_table
    
    
    def minimum_ols(self, X, y):
        n, p = X.shape
        dfe = n - p - 1.0
        dft = n - 1.0
        dfr = p
        
        sst = np.var(y, axis=0)*y.shape[0]
        
        G = pinv(X.T.dot(X))
        beta =  G.dot(X.T.dot(y))
        yhat = _check_np(X.dot(beta))
        
        ssr = np.sum((yhat - _check_np(np.mean(y, axis=0)))**2, axis=0)
        sse = np.sum((yhat - _check_np(y))**2, axis=0)
        
        res = [sst, ssr, sse, sst/dft, ssr/dfr, sse/dfe, ssr/sst, 1.0 - (sse/dfe)/(sst/dft)]
        res = [_check_np(_check_1d(x)) for x in res]
        return res
    
    def loglike(self, coefs, error_var):
        n = self.X.shape[0]
        k = n/2.0 * np.log(2*np.pi)
        ldt = n/2.0*np.log(error_var)
        dev = 0.5
        ll = k + ldt + dev
        return ll

    

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
        Currently only supports canonical links for binomial, poisson,
        and gamma distributions.
        
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
        elif family.lower() in ['gamma', 'gam']:
            self.family = GammaGLM()
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
    
    
class GammaGLM:
    
    def __init__(self):
        self.dist = 'g'
    
    def cumulant(self, Xb):
        bt = -np.log(-Xb)
        return bt
    
    def cumulant_prime(self, Xb):
        db = -1/Xb
        return db
    
    def cumulant_double_prime(self, Xb):
        d2b = 1/(Xb**2)
        return d2b



