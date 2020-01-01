#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:33:37 2019

@author: lukepinkel
"""


import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import scipy.optimize # analysis:ignore
import scipy.stats# analysis:ignore
import scipy.special# analysis:ignore
import patsy  # analysis:ignore
import pandas as pd # analysis:ignore
from .utils import linalg_utils, base_utils # analysis:ignore

LN2PI = np.log(2.0 * np.pi)
FOUR_SQRT2 = 4.0 * np.sqrt(2.0)



class GLM:
    
    def __init__(self, frm=None, data=None, fam=None, scale_estimator='M'):
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
        self.theta_init = np.zeros(self.X.shape[1])
        if isinstance(fam, Gamma):
            self.theta_init = np.linalg.lstsq(self.X, self.f.link(self.Y))[0]
            self.theta_init = linalg_utils._check_1d(self.theta_init)
        
        
        if isinstance(fam, (Binomial, Poisson)):
            self.scale_handling = 'fixed'
        else:
            self.scale_handling = scale_estimator   
        
        if self.scale_handling == 'NR':
            if isinstance(fam, Gamma):
                phi_init = self._est_scale(self.Y, 
                                           self.f.inv_link(self.X.dot(self.theta_init)),
                                           )
            else:
                phi_init = np.ones(1)
            self.theta_init = np.concatenate([self.theta_init, np.atleast_1d(phi_init)])
     
    def _est_scale(self, y, mu):
        y, mu = self.f.cshape(y, mu)
        r = (y - mu)**2
        v = self.f.var_func(mu=mu)
        s = np.sum(r / v)
        s/= self.dfe
        return s
    
    def loglike(self, params):
        params = linalg_utils._check_1d(params)
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        ll = self.f.loglike(self.Y, mu=mu, scale=phi)
        return ll

    def gradient(self, params):
        params = linalg_utils._check_1d(params)
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
            dt = np.atleast_1d(np.sum(self.f.dtau(tau, self.Y, mu)))
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        w = self.f.gw(self.Y, mu=mu, phi=phi)
        g = np.dot(self.X.T, w)
        if self.scale_handling == 'NR':
            g = np.concatenate([g, dt])
        return g
    
    def hessian(self, params):
        if self.scale_handling == 'NR':
            beta, tau = params[:-1], params[-1]
            eta = self.X.dot(beta)
            mu = self.f.inv_link(eta)
            phi = np.exp(tau)
            d2t = np.atleast_2d(self.f.d2tau(tau, self.Y, mu))
            dbdt = -np.atleast_2d(self.gradient(params)[:-1])
        else:
            eta = self.X.dot(params)
            mu = self.f.inv_link(eta)
            if self.scale_handling == 'M':
                phi = self._est_scale(self.Y, mu)
            else:
                phi = 1.0
        w = self.f.hw(self.Y, mu=mu, phi=phi)
        H = (self.X.T * w).dot(self.X)
        if self.scale_handling == 'NR':
            H = np.block([[H, dbdt.T], [dbdt, d2t]])
        return H 
    
    
    def _fit_optim(self):
        opts = {'verbose':3}
        optimizer = sp.optimize.minimize(self.loglike, self.theta_init,
                                         jac=self.gradient,
                                         hess=self.hessian, options=opts,
                                         method='trust-constr')
        return optimizer
    
    def _fit_manual(self, theta=None):
        if theta is None:
            theta = self.theta_init
    
        fit_hist = {'|g|':[], 'theta':[], 'i':[]}
        ll_k = self.loglike(theta)
        sh = 1.0
        for i in range(100): 
            H = self.hessian(theta)
            g =  self.gradient(theta)
            gnorm = np.linalg.norm(g)
            fit_hist['|g|'].append(gnorm)
            fit_hist['i'].append(i)
            fit_hist['theta'].append(theta)
            if gnorm/len(g)<1e-9:
                 break
            dx = np.atleast_1d(np.linalg.solve(H, g))
            if self.loglike(theta - dx)>ll_k:
                for j in range(100):
                    sh*=2
                    if self.loglike(theta - dx/sh)<ll_k:
                        break
            theta -= dx/sh
            sh = 1.0
        return theta, fit_hist
            
    
class Link(object):
    def inv_link(self, eta):
        raise NotImplementedError
    
    def dinv_link(self, eta):
        raise NotImplementedError
        
    def d2inv_link(self, eta):
        raise NotImplementedError
    
    def link(self, mu):
        raise NotImplementedError
    
    def dlink(self, mu):
        raise NotImplementedError
        

class IdentityLink(Link):

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
    
    def link(self, mu):
        return mu
    
    def dlink(self, mu):
        return 1
    
    
class LogitLink(Link):

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
        
        

class ProbitLink(Link):
    
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
        
        

class LogLink(Link):
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
    
    
class ReciprocalLink(Link):
    
    
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


class CloglogLink(Link):
    
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
    
    

class PowerLink(Link):
    
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
    
    
    

def _logbinom(n, k):
    y=sp.special.gammaln(n+1)-sp.special.gammaln(k+1)-sp.special.gammaln(n-k+1)
    return y

class ExponentialFamily(object):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        
        if not isinstance(link, Link):
            link = link()

        self._link = link
        self.weights = weights
        self.scale = scale
        
        
    def _to_mean(self, eta=None, T=None):
        if eta is not None:
            mu = self.inv_link(eta)
        else:
            mu = self.mean_func(T)
        return mu   
    
    def link(self, mu):
        return self._link.link(mu)

    def inv_link(self, eta):
        return self._link.inv_link(eta)
        
    def dinv_link(self, eta):
        return self._link.dinv_link(eta)
    
    def d2inv_link(self, eta):
        return self._link.d2inv_link(eta)
    
    def dlink(self, mu):
        return 1.0 / self.dinv_link(self.link(mu))
    
    def d2link(self, mu):
        eta = self.link.link(mu)
        res = -self.d2inv_link(eta) / np.power(self.dinv_link(eta), 3)
        return res
    
    def cshape(self, y, mu):
        y = linalg_utils._check_1d(linalg_utils._check_np(y))
        mu = linalg_utils._check_1d(linalg_utils._check_np(mu))
        return y, mu
    
    def loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        return np.sum(self._loglike(y, eta, mu, T, scale, dispersion=1.0))
    
    def full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0):
        return np.sum(self._full_loglike(y, eta, mu, T, scale))
    
    def pearson_resid(self, y, eta=None, mu=None, T=None, scale=1.0, phi=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        V = self.var_func(mu)
        r_p = (y - mu) / np.sqrt(V)
        return r_p
    
    def signed_resid(self, y, eta=None, mu=None, T=None, scale=1.0, phi=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        y, mu = self.cshape(y, mu)
        d = self.deviance(y, mu=mu)
        r_s = np.sign(y - mu) * np.sqrt(d)
        return r_s
    
    def gw(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        num = self.weights * (y - mu)
        den = self.var_func(mu=mu) * self.dlink(mu) * phi
        res = num / den
        return -res
    
    def hw(self, y, mu, phi=1.0):
        y, mu = self.cshape(y, mu)
        eta = self.link(mu)
        Vinv = 1.0 / (self.var_func(mu=mu))
        W0 = self.dinv_link(eta)**2
        W1 = self.d2inv_link(eta)
        W2 = self.d2canonical(mu)
        
        
        Psc = (y-mu) * (W2*W0+W1*Vinv)
        Psb = Vinv*W0
        res = (Psc - Psb)*self.weights
        return -res/phi
        
        

        
                       
    
    
    
        
        

class Gaussian(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
            
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        
        ll= w * np.power((y - mu), 2) + np.log(scale/self.weights)
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + LN2PI
        return llf
    
    def canonical_parameter(self, mu):
        T = mu
        return T
    
    def cumulant(self, T):
        b = T**2  / 2.0
        return b
    
    def mean_func(self, T):
        mu = T
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        V = mu*0.0+1.0
        return V
                
    def d2canonical(self, mu):
        res = 0.0*mu+1.0
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        d = w * np.power((y - mu), 2.0)
        return d
    
    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = -np.sum(w * np.power((y - mu), 2) / phi - 1)
        return g
    
    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = np.sum(w * np.power((y - mu), 2) / (2 * phi))
        return g
        
    
    

     

class InverseGaussian(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0, dispersion=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
           
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        
        ll = w * np.power((y - mu), 2) / (y * mu**2)
        ll+= np.log((scale * y**2) / self.weights)
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + LN2PI
        return llf 

    
    def canonical_parameter(self, mu):
        T = 1.0 / (np.power(mu, 2.0))
        return T
    
    def cumulant(self, T):
        b = -np.sqrt(-2.0*T)
        return b
    
    def mean_func(self, T):
        mu = 1.0 / np.sqrt(-2.0*T)
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
    
        V = np.power(mu, 3.0)
        return V
                
    def d2canonical(self, mu):
        res = 3.0 / (FOUR_SQRT2 * np.power(-mu, 2.5))
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        d = w * np.power((y - mu), 2.0) / (y * np.power(mu, 2))
        return d
    
    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        num = w * np.power((y - mu), 2)
        den = (phi * y * np.power(mu, 2))
        g = -np.sum(num / den - 1)
        return g    
    
    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        g = np.sum(w * np.power((y - mu), 2) / (2 * phi * y * mu**2))
        return g


class Gamma(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0, dispersion=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
           
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        z = w * y / mu
        ll = z - w * np.log(z) + sp.special.gammaln(self.weights/scale)
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + np.log(y)
        return llf 

    
    def canonical_parameter(self, mu):
        T = -1.0 / mu
        return T
    
    def cumulant(self, T):
        b = -np.log(-T)
        return b
    
    def mean_func(self, T):
        mu = -1 / T
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
    
        V = linalg_utils._check_1d(mu)**2
        return V
                
    def d2canonical(self, mu):
        res = -2 /(mu**3)
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        d = 2 * w * ((y - mu) / mu - np.log(y / mu))
        return d
    
    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (1 - y / mu)
        T2 = -sp.special.digamma(w / phi)
        g = (w / phi) * (T0 + T1 + T2)
        return g 
    
    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        T0 = np.log(w * y / (phi * mu))
        T1 = (2 - y / mu)
        T2 = sp.special.digamma(w / phi)
        T3 = w / phi * sp.special.polygamma(1, w / phi)
        g = np.sum(w / phi * (T3+T2-T1-T0))
        return g

    

    

class NegativeBinomial(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
           
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        
        v = 1.0 / dispersion
        kmu = dispersion*mu
        
        yv = y + v
        
        ll = yv * np.log(1.0 + kmu) - y * np.log(kmu)
        ll+= sp.special.gammaln(v) - sp.special.gammaln(yv)
        ll*= w
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + self.weights / scale * sp.special.gammaln(y + 1.0)
        return llf 

    
    def canonical_parameter(self, mu, dispersion=1.0):
        u = mu * dispersion
        T = np.log(u / (1.0 + u))
        return T
    
    def cumulant(self, T, dispersion=1.0):
        b = (-1.0 / dispersion) * np.log(1 - dispersion * np.exp(T))
        return b
    
    def mean_func(self, T, dispersion=1.0):
        u = np.exp(T)
        mu = -1.0 / dispersion * (u / (1 - u))
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
    
        V = mu + np.power(mu, 2) * dispersion
        return V
                
    def d2canonical(self, mu, dispersion=1.0):
        res = -2 * dispersion * mu - 1
        res/= (np.power(mu, 2) * np.power((mu*dispersion+1.0), 2))
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        d = np.zeros(y.shape[0])
        ix = (y==0)
        v = 1.0 / dispersion
        d[ix] = np.log(1 + dispersion * mu[ix]) / dispersion
        yb, mb = y[~ix], mu[~ix]
        u = (yb + v) / (mb + v)
        d[~ix] =  (yb*np.log(yb / mb) - (yb + v) * np.log(u))
        d *= 2*w
        return d
    
    def dtau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        A = phi * (y - mu) / ((1 + phi) * mu)
        T0 = sp.special.digamma(y + 1 / phi)
        T1 = np.log(1+phi*mu)
        T2 = sp.special.digamma(1 / phi)
        g = (w / phi) * (T0 - T1 - T2 - A)
        return g 
    
    def d2tau(self, tau, y, mu):
        y, mu = self.cshape(y, mu)
        w = self.weights
        phi = np.exp(tau)
        v = 1/phi
        T0 = v*np.log(1+phi*mu)
        T1 = v*(sp.special.digamma(y+v) - sp.special.digamma(v))
        T2 = v**2 * (sp.special.polygamma(2, y+v)-sp.special.polygamma(2, v))
        A = -y*phi*mu+mu+2*phi*mu**2 / ((1+phi*mu)**2)
        g = np.sum(w / phi * (T0 - A - T1 - T2))
        return g

    


    

    
    
class Poisson(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
           
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        
        ll = -w * (y * np.log(mu) - mu)
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        llf = ll + self.weights / scale * np.log(sp.special.factorial(y))
        return llf 

    
    def canonical_parameter(self, mu, dispersion=1.0):
        T = np.log(mu)
        return T
    
    def cumulant(self, T, dispersion=1.0):
        b = np.exp(T)
        return b
    
    def mean_func(self, T, dispersion=1.0):
        mu = np.exp(T)
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
    
        V = mu
        return V
                
    def d2canonical(self, mu, dispersion=1.0):
        res = -1  /(mu**2)
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        d = np.zeros(y.shape[0])
        ixa = y==0
        ixb = ~ixa
        d[ixa] = mu[ixa]
        d[ixb] = (y[ixb]*np.log(y[ixb]/mu[ixb]) - (y[ixb] - mu[ixb]))
        d*=2.0 * w
        return d
    

    
    
class Binomial(ExponentialFamily):
    
    def __init__(self, link=IdentityLink, weights=1.0, scale=1.0):
        super().__init__(link, weights, scale)
    
 
    def _loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
           
        y, mu = self.cshape(y, mu)
        w = self.weights / scale
        
        ll = -w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))
        return ll
    
    def _full_loglike(self, y, eta=None, mu=None, T=None, scale=1.0, dispersion=1.0):
        ll = self._loglike(y, eta, mu, T, scale)
        w = self.weights
        r = w * y
        llf = ll - _logbinom(w, r)
        return llf 

    
    def canonical_parameter(self, mu, dispersion=1.0):
        u = mu / (1  - mu)
        T = np.log(u)
        return T
    
    def cumulant(self, T, dispersion=1.0):
        u = 1 + np.exp(T)
        b = np.log(u)
        return b
    
    def mean_func(self, T, dispersion=1.0):
        u = np.exp(T)
        mu = u / (1 + u)
        return mu
    
    def var_func(self, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
    
        V = mu * (1 - mu)
        return V
                
    def d2canonical(self, mu, dispersion=1.0):
        res = 1.0/((1 - mu)**2)-1.0/(mu**2)
        return res
    
    def deviance(self, y, T=None, mu=None, eta=None, scale=1.0, dispersion=1.0):
        if mu is None:
            mu = self._to_mean(eta=eta, T=T)
        
        y, mu = self.cshape(y, mu)
        w = self.weights
        ixa = y==0
        ixb = (y!=0)&(y!=1)
        ixc = y==1
        d = np.zeros(y.shape[0])
        u = (1 - y)[ixb]
        v = (1 - mu)[ixb]
        d[ixa] = -np.log(1-mu[ixa])
        d[ixc] = -np.log(mu[ixc])
        d[ixb] = y[ixb]*np.log(y[ixb]/mu[ixb]) + u*np.log(u/v)
        return 2*w*d
    

    
    
    
    
    
            
        
    
    
    
    
    
    
    

