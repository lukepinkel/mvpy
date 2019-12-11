#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 18:08:10 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize 
import scipy.stats
import scipy.special
import patsy
from ..utils import base_utils
from ..utils import linalg_utils


class MinimalGLM:
    
    def __init__(self, X, Y, fam):
        self.f = fam
        self.X, self.xcols, self.xix, self.x_is_pd = base_utils.check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = base_utils.check_type(Y)
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
        return -linalg_utils._check_0d(LL)
    
    def gradient(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv =1.0/V
        W = Vinv * self.f.dinv_link(eta)
        G = -(linalg_utils._check_1d(Y) - mu) * W
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
        
        Psc = (linalg_utils._check_1d(Y)-mu) * (W2 * W0 + W1 * Vinv)
        Psb = Vinv*W0
        W = Psc - Psb
        
        H = (X.T * W).dot(X)
        return -H
    
    def fit(self, verbose=2):
        theta = np.ones(self.X.shape[1])/2.0
        if self.f.dist == 'normal':
            theta *= 1e-10
        optimizer = sp.optimize.minimize(self.loglike, theta, jac=self.gradient, 
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
        Y, X = patsy.dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = base_utils.check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = base_utils.check_type(Y)
        self.n_obs, self.n_feats = self.X.shape
        self.dfe = self.n_obs - self.n_feats
        self.jn = np.ones((self.n_obs, 1))
        self.YtX = self.Y.T.dot(self.X)
        self.theta_init = np.ones(self.n_feats)/self.n_feats
        if self.f.dist == 'normal':
            self.theta_init *= 1e-10
        
    def loglike(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y, jn = self.X, self.Y, self.jn
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        Z = self.f.cumulant(T)
        LL = (Y.T.dot(T) - jn.T.dot(Z)) / phi
        return -linalg_utils._check_0d(LL)
    
    def gradient(self, params):
        beta, phi = self.f.unpack_params(params)
        X, Y = self.X, self.Y
        eta = X.dot(beta)
        mu = self.f.inv_link(eta)
        T = self.f.canonical_parameter(mu)
        V = self.f.var_func(T)
        Vinv =1.0/V
        W = Vinv * self.f.dinv_link(eta) * self.f.weights
        G = -(linalg_utils._check_1d(Y) - mu) / phi * W
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
        
        
        Psc = (linalg_utils._check_1d(Y)-mu) * (W2*W0+W1*Vinv)
        Psb = Vinv*W0
        W = (Psc - Psb)*self.f.weights
        
        H = (X.T * W).dot(X) / phi
        return -H
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr',
                                'options':{'verbose':0}}
        X0 = np.ones((self.n_obs, 1))
        intercept_model = MinimalGLM(X0, self.Y, self.f)
        intercept_model.fit()
        self.intercept_model = intercept_model
        self.LL0 = intercept_model.loglike(intercept_model.beta)
        theta = self.theta_init
        optimizer = sp.optimize.minimize(self.loglike, theta, jac=self.gradient, 
                             hess=self.hessian, **optimizer_kwargs)        
        self.optimizer = optimizer
        self.beta = optimizer.x
        self.sse = np.sum((self.Y[:, 0]-self.predict())**2)
        if self.f.dist == 'normal':
            self.f.phi = self.sse / (self.n_obs - self.X.shape[1])
        self.hess = self.hessian(self.beta)
        self.grad = self.gradient(self.beta)
        self.vcov = linalg_utils.einv(self.hess)
        self.beta_se = np.sqrt(np.diag(self.vcov))
        self.tvals = self.beta/self.beta_se
        self.pvals = sp.stats.t.sf(abs(self.tvals), self.dfe)*2.0
        self.res = pd.DataFrame(np.vstack([self.beta, self.beta_se,
                                           self.tvals, self.pvals]).T, 
                                           index=self.xcols, 
                                           columns=['beta', 'SE', 't', 'p'])

        self.LLA = self.loglike(self.beta)
        if  self.f.dist == 'poisson':
            self.LLA = -self.LLA
            self.LL0 = -self.LL0
        self.LLR = 2.0*(self.LL0-self.LLA)
    
        self.sst = np.var(self.Y)*self.Y.shape[0]
        n, p = self.X.shape[0], self.X.shape[1]
        yhat = self.predict()
        y = linalg_utils._check_1d(self.Y)
        self.var_mu = self.f.var_func(self.f.canonical_parameter(yhat)).mean()
       
        self.ssr =np.sum((yhat - yhat.mean())**2)
        pt = y.mean()
        if self.f.dist in ['binomial', 'bernoulli']:
            rmax =  (1 - np.exp(-2.0/n * (self.LL0)))
            rmax_an = -2*(pt*np.log(pt) + (1-pt)*np.log(1-pt))
            rmax_an = rmax_an / (1 + rmax_an)
        else:
            rmax = 2*self.LL0/(self.n_obs+2*self.LL0)
            rmax_an = rmax
     
        mu_p = yhat.mean()
        pseudo_r2 = {}
        pseudo_r2['Lave-Efron'] = 1 - self.sse / self.sst
        pseudo_r2['Cox-Snell'] = 1 - np.exp(2.0/n*(self.LLA-self.LL0))
        pseudo_r2['Nagelkerke'] = pseudo_r2['Cox-Snell'] / rmax
        pseudo_r2['McFaddens'] = 1 - self.LLA/self.LL0
        pseudo_r2['McFaddens_adj'] = 1 - (self.LLA-p)/self.LL0
        pseudo_r2['McKelvey'] = self.ssr/(self.ssr+3.29*n)
        pseudo_r2['Aldrich'] = self.LLR/(self.LLR+n)
        pseudo_r2['Veall-Zimmerman'] = pseudo_r2['Aldrich'] / rmax_an
        pseudo_r2['Tjur-mod'] = ((yhat - mu_p)**2).sum()/(n*mu_p*(1 - mu_p))
        pseudo_r2['Tjur-res'] = 1 - (yhat * (1 - yhat)).sum() / (n*mu_p*(1 - mu_p))
        pseudo_r2['D'] = 0.5 * (pseudo_r2['Tjur-mod'] + pseudo_r2['Tjur-res'])
        self.pseudo_r2 = pseudo_r2
        self.LLRp =  sp.stats.chi2.sf(self.LLR, len(self.beta))
        self.pearson_chi2 = self.sse / self.var_mu
        self.deviance = self.f.deviance(self.optimizer.x, self.X, self.Y)
        self.scale_chi2 = self.pearson_chi2 / (n-p)
        self.scale_dev = self.deviance.sum() / (n - p)
        self.pearson_resid = (y - yhat)*np.sqrt(1/self.var_mu)
        self.AIC = 2*self.LLA + 2*p
        self.AICC = 2*self.LLA + 2*p*n/(n-p-1)
        self.BIC = 2*self.LLA + p*np.log(n)
        ix = ['LL Model', 'LL Intercept', 'LL Ratio', 'LLR p value', 
              'Pearson chi2', 'Deviance', 'AIC', 'AICC', 'BIC']
        self.sumstats = pd.DataFrame(
                [self.LLA, self.LL0, self.LLR, self.LLRp, self.pearson_chi2, 
                 self.deviance.sum(), self.AIC, self.AICC, self.BIC], index=ix,
                 columns=['Value'])
        self.sumstats = pd.concat([self.sumstats, pd.DataFrame(self.pseudo_r2, 
                                                index=['Value']).T],axis=0)
        
        
        
     
    def predict(self, X=None):
        if X is None:
            X = self.X
        return self.f.inv_link(X.dot(self.beta))
    
    
class Bernoulli:
    
    def __init__(self, link='canonical'):
        self.dist = 'bernoulli'
        if link == 'canonical':
            self.link=LogitLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
        self.weights = 1.0
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
        V =linalg_utils. _check_1d(mu * (1 - mu))
        return V
                
    def d2canonical(self, mu):
        res = 1.0/((1.0 - mu)**2)-1.0/(mu**2)
        return res
    
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    
    def deviance(self, params, X, Y):
        y = linalg_utils. _check_1d(Y)
        mu = self.inv_link(X.dot(params))
        lna, lnb = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        ixa, ixb = (y/mu)>0, ((1-y)/(1-mu))>0
        lna[ixa] = np.log(y[ixa]/mu[ixa]) 
        lnb[ixb] = np.log((1-y[ixb])/(1-mu[ixb]))
        d = y*lna+(1-y)*lnb
        return 2*d
    

class Binomial:
    
    def __init__(self, weights=None, link='canonical'):
        if weights is None:
            self.weights = np.ones(1)
        else:
            self.weights = weights
        self.dist = 'binomial'
        if link == 'canonical':
            self.link=LogitLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
            
    def canonical_parameter(self, mu):
        u = mu / (1.0  - mu)
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
        V = linalg_utils. _check_1d(mu * (1 - mu))
        return V
                
    def d2canonical(self, mu):
        res = 1.0/((1 - mu)**2)-1.0/(mu**2)
        return res
    
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    
    def deviance(self, params, X, Y):
        y = linalg_utils. _check_1d(Y)
        mu = self.inv_link(X.dot(params))
        lna, lnb = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        ixa, ixb = (y/mu)>0, ((1-y)/(1-mu))>0
        lna[ixa] = np.log(y[ixa]/mu[ixa]) 
        lnb[ixb] = np.log((1-y[ixb])/(1-mu[ixb]))
        d = y*lna+(1-y)*lnb
        return 2*d
    

        

class Poisson:
    
    def __init__(self, link='canonical'):
        self.dist = 'poisson'
        if link == 'canonical':
            self.link=LogLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
        self.weights = 1.0
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
        V = linalg_utils._check_1d(mu)
        return V
                
    def d2canonical(self, mu):
        res = -1  /(mu**2)
        return res
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    
    def deviance(self, params, X, Y):
        y = linalg_utils._check_1d(Y)
        mu = self.inv_link(X.dot(params))
        d = np.zeros(y.shape[0])
        ix = (y==0)
        d[~ix] = (y[~ix]*np.log(y[~ix]/mu[~ix]) - (y[~ix] - mu[~ix]))
        d[ix]= mu[ix]
        return 2*d
    

class Gamma:
    
    def __init__(self, link='canonical'):
        self.dist = 'gamma'
        if link == 'canonical':
            self.link=ReciprocalLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
        self.weights = 1.0
    def canonical_parameter(self, mu):
        T = 1.0 / mu
        return T
    
    def inv_link(self, eta):
        return self.link.inv_link(eta)
        
    def dinv_link(self, eta):
        return self.link.dinv_link(eta)
    
    def d2inv_link(self, eta):
        return self.link.d2inv_link(eta)
    
    def cumulant(self, T):
        b = -np.log(-T)
        return b
    
    def mean_func(self, T):
        mu = -1 / T
        return mu
    
    def var_func(self, T):
        mu = self.mean_func(T)
        V = linalg_utils._check_1d(mu)**2
        return V
                
    def d2canonical(self, mu):
        res = 2 /(mu**3)
        return res
    
    def unpack_params(self, params):
        beta = params
        phi = 1.0
        return beta, phi
    
    def deviance(self, params, X, Y):
        y = linalg_utils._check_1d(Y)
        mu = self.inv_link(X.dot(params))
        lna, lb = np.zeros(y.shape[0]), np.zeros(y.shape[0])
        ixa, ixb = (y/mu)>0, mu!=0
        lna[ixa] = np.log(y[ixa]/mu[ixa]) 
        lb[ixb] = (y - mu) / mu
        d = lb - lna
        return 2*d


class Normal:
    
    def __init__(self, phi, link='canonical'):
        self.dist = 'normal'
        self.phi = phi
        if link == 'canonical':
            self.link=IdentityLink()
            self.type_='canonical'
        else:
            self.link = link
            self.type_='noncanonical'
        self.weights = 1.0
    def canonical_parameter(self, mu):
        T = mu
        return T
    
    def inv_link(self, eta):
        return self.link.inv_link(eta)
        
    def dinv_link(self, eta):
        return self.link.dinv_link(eta)
    
    def d2inv_link(self, eta):
        return self.link.d2inv_link(eta)
    
    def cumulant(self, T):
        b = T**2  / 2.0
        return b
    
    def mean_func(self, T):
        mu = T
        return mu
    
    def var_func(self, T):
        V = T*0.0+1.0
        return V
                
    def d2canonical(self, mu):
        res = 0.0*mu+1.0
        return res
    
    def unpack_params(self, params):
        beta = params
        phi = self.phi
        return beta, phi
    
    def deviance(self, params, X, Y):
        y = linalg_utils._check_1d(Y)
        mu = self.inv_link(X.dot(params))
        d = (y - mu)**2
        return 2*d


class IdentityLink:

    def __init__(self):
        self.fnc='identity'
        
    def inv_link(self, eta):
        mu = eta
        return mu
    
    def dinv_link(self, eta):
        dmu = 0.0*eta+1.0
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = 0.0*eta
        return d2mu
    
    
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
    
    def link(self, mu):
        eta = np.log(mu / (1 - mu))
        return eta
    
    def dlink(self, mu):
        dmu = 1 / (mu * (1 - mu))
        return dmu
        
        

class ProbitLink:
    
    def __init__(self):
        self.fnc='probit'
        
    def inv_link(self, eta):
        mu = sp.stats.norm.cdf(eta, loc=0, scale=1)
        mu[mu==1.0]-=1e-16
        return mu
    
    def dinv_link(self, eta):
        dmu = sp.stats.norm.pdf(eta, loc=0, scale=1)
        return dmu
        
    def d2inv_link(self, eta):
        d2mu = -eta * sp.stats.norm.pdf(eta, loc=0, scale=1)
        return d2mu
    
    def link(self, mu):
        eta = sp.stats.norm.ppf(mu, loc=0, scale=1)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
        
        

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
    
    def link(self, mu):
        eta = np.log(mu)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
    
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
    
    def link(self, mu):
        eta  = 1 / mu
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu


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
    
    def link(self, mu):
        eta = np.log(np.log(1 / (1 - mu)))
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
    

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
    
    def link(self, mu):
        if self.alpha==0:
            eta = np.log(mu)
        else:
            eta = mu**(self.alpha)
        return eta
    
    def dlink(self, mu):
        dmu = 1.0 / (self.dinv_link(self.link(mu)))
        return dmu
    
    
    

    
    
    
