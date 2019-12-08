#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:41:56 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize
import patsy
from numpy import exp, ones, dot, diag

from ..utils import linalg_utils, base_utils, statfunc_utils


     
        
class MinimalCLM:
    
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
        self.n_cats = len(np.unique(self.Y[~np.isnan(self.Y)]))
        self.resps = np.unique(self.Y[~np.isnan(self.Y)])
        self.resps = np.sort(self.resps)
        #self.W = self.Y.dot(np.arange(self.n_cats))+1.0
        self.W = ones(self.X.shape[0])
        self.constraints = [dict(zip(
                ['type', 'fun'], 
                ['ineq', lambda params: params[i+1]-params[i]])) 
                for i in range(self.n_cats-2)]
        self.A1, self.A2 = self.Y[:, :-1], self.Y[:, 1:]
        self.o1, self.o2 = self.Y[:, -1]*10e1, self.Y[:, 0]*-10e5
        self.B1, self.B2 = np.block([self.A1, -self.X]), np.block([self.A2, -self.X])
        

    def inv_logit(self, x):
        y = exp(x) / (1.0 + exp(x))
        return y
    
    def dlogit(self, x):
        y = exp(x) / ((1.0 + exp(x))**2)
        return y
    
    def d2logit(self, x):
        y = (exp(x)*(1.0 - exp(x))) / (exp(x) + 1.0)**3.0
        return y
    
    
    def loglike(self, params):
        W = self.W
        params = linalg_utils._check_1d(params)
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        Pi = Gamma_1 - Gamma_2
        LL = np.sum(W * np.log(Pi))
        return -LL
    
    def gradient(self, params):
        W = self.W
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        dPi = (B1 * Phi_11[:, None]).T - (B2*Phi_12[:, None]).T
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        
        g = -dot(dPi, W / Pi)
        return g
    
    
    def hessian(self, params):
        W = self.W
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Phi_21, Phi_22 = self.d2logit(Nu_1),  self.d2logit(Nu_2)
        #Phi_21[Y[:, 0].astype(bool)], Phi_22[Y[:, -1].astype(bool)] = 1.0, 0.0
    
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        Phi_11 = linalg_utils._check_2d(Phi_11)
        Phi_12 = linalg_utils._check_2d(Phi_12)
        Phi_21 = linalg_utils._check_2d(Phi_21)
        Phi_22 = linalg_utils._check_2d(Phi_22)
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        Phi3 =linalg_utils._check_2d(W / Pi**2)
        dPi = (B1 * Phi_11).T - (B2*Phi_12).T
        T0 = (B1 * Phi_21).T.dot(B1)
        T1 = (B2 * Phi_22).T.dot(B2)
        T2 = dPi.dot(dPi.T*Phi3)
        H=T0-T1-T2
        return -H
    
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr',
                                'options':{'verbose':0}}
            
        
        theta = statfunc_utils.norm_qtf(np.sum(self.Y, axis=0).cumsum()[:-1]/np.sum(self.Y))
        beta = ones(self.X.shape[1])
        params = np.concatenate([theta, beta], axis=0)
        self.theta_init = theta
        self.params_init = params

        res = sp.optimize.minimize(self.loglike, params, constraints=self.constraints,
                       jac=self.gradient, hess=self.hessian, **optimizer_kwargs)
        

        self.params = res.x
        self.optimizer = res
        

class CLM:
    
    def __init__(self, frm, data, X=None, Y=None):
        Y, X = patsy.dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = base_utils.check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_p = base_utils.check_type(Y)
        self.Y = linalg_utils._check_1d(self.Y)
        self.n_cats = len(np.unique(self.Y[~np.isnan(self.Y)]))
        self.resps = np.unique(self.Y[~np.isnan(self.Y)])
        self.resps = np.sort(self.resps)
        self.Y = np.concatenate([(self.Y==x)[:, None] for x in self.resps],
                                 axis=1).astype(float)
        #self.W = self.Y.dot(np.arange(self.n_cats))+1.0
        self.W = ones(self.X.shape[0])
        self.constraints = [dict(zip(
                ['type', 'fun'], 
                ['ineq', lambda params: params[i+1]-params[i]])) 
                for i in range(self.n_cats-2)]
        self.A1, self.A2 = self.Y[:, :-1], self.Y[:, 1:]
        self.o1, self.o2 = self.Y[:, -1]*10e1, self.Y[:, 0]*-10e5
        self.B1, self.B2 = np.block([self.A1, -self.X]), np.block([self.A2, -self.X])
        

    def inv_logit(self, x):
        y = exp(x) / (1.0 + exp(x))
        return y
    
    def dlogit(self, x):
        y = exp(x) / ((1.0 + exp(x))**2)
        return y
    
    def d2logit(self, x):
        y = (exp(x)*(1.0 - exp(x))) / (exp(x) + 1.0)**3.0
        return y
    
    
    def loglike(self, params):
        W = self.W
        params = linalg_utils._check_1d(params)
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        Pi = Gamma_1 - Gamma_2
        LL = np.sum(W * np.log(Pi))
        return -LL
    
    def gradient(self, params):
        W = self.W
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        dPi = (B1 * Phi_11[:, None]).T - (B2*Phi_12[:, None]).T
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        
        g = -dot(dPi, W / Pi)
        return g
    
    def hessian(self, params):
        W = self.W
        o1, o2 = self.o1, self.o2
        B1, B2 = self.B1, self.B2
        Nu_1, Nu_2 = B1.dot(params)+o1, B2.dot(params)+o2
        Phi_21, Phi_22 = self.d2logit(Nu_1),  self.d2logit(Nu_2)
        #Phi_21[Y[:, 0].astype(bool)], Phi_22[Y[:, -1].astype(bool)] = 1.0, 0.0
    
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        Phi_11 = linalg_utils._check_2d(Phi_11)
        Phi_12 = linalg_utils._check_2d(Phi_12)
        Phi_21 = linalg_utils._check_2d(Phi_21)
        Phi_22 = linalg_utils._check_2d(Phi_22)
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        Phi3 =linalg_utils._check_2d(W / Pi**2)
        dPi = (B1 * Phi_11).T - (B2*Phi_12).T
        T0 = (B1 * Phi_21).T.dot(B1)
        T1 = (B2 * Phi_22).T.dot(B2)
        T2 = dPi.dot(dPi.T*Phi3)
        H=T0-T1-T2
        return -H
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr',
                                'options':{'verbose':0}}
            
        intercept_model = MinimalCLM(np.ones((self.X.shape[0], 1)), self.Y)
        intercept_model.fit()
        self.intercept_model = intercept_model
        theta = statfunc_utils.norm_qtf(np.sum(self.Y, axis=0).cumsum()[:-1]/np.sum(self.Y))
        beta = ones(self.X.shape[1])*0
        params = np.concatenate([theta, beta], axis=0)
        self.theta_init = theta
        self.params_init = params

        res = sp.optimize.minimize(self.loglike, params, constraints=self.constraints,
                       jac=self.gradient, hess=self.hessian, **optimizer_kwargs)
        

        self.params = res.x
        self.optimizer = res
        self.H = self.hessian(self.params)
        self.Vcov = np.linalg.pinv(self.H)
        self.SE = diag(self.Vcov)**0.5
        self.res = np.concatenate([self.params[:, None],
                                   self.SE[:, None]], axis=1)
        self.res = pd.DataFrame(self.res, columns=['coef', 'SE'])
        self.res['t'] = self.res['coef']/self.res['SE']
        idx = ["threshold %i|%i"%(i, i+1) for i in range(1, len(theta)+1)]
        if self.xcols is not None:
          idx = idx+self.xcols.tolist()
        else:
          idx = idx+["beta%i"%i for i in range(self.X.shape[1])]
        self.res.index  = idx
        self.res['p']  = sp.stats.t.sf(np.abs(self.res['t']), self.Y.shape[0]-len(self.params))*2.0
        self.theta = self.params[:len(theta)]
        self.beta = self.params[len(theta):]
        self.LLA = self.loglike(self.params)
        self.LL0 = self.intercept_model.loglike(self.intercept_model.params)
        self.LLR = self.LL0 - self.LLA
        self.LLRp = sp.stats.chi2.sf(self.LLR, 
                                  len(self.params) - len(self.intercept_model.params))
        n, p = self.X.shape
        rmax =  (1 - np.exp(-2.0/n * (self.LL0)))
        r2_coxsnell = 1 - np.exp(2.0/n*(self.LLA-self.LL0))
        r2_mcfadden = 1 - self.LLA/self.LL0
        r2_mcfadden_adj = 1 - (self.LLA-p)/self.LL0
        r2_nagelkerke = r2_coxsnell / rmax
        ss = [[self.LLR, self.LLRp],
              [r2_coxsnell, '-'],
              [r2_mcfadden, '_'],
              [r2_mcfadden_adj, '-'],
              [r2_nagelkerke, '-']]
        ss = pd.DataFrame(ss, index=['LLR', 'r2_coxsnell', 'r2_mcfadden',
                                     'r2_mfadden_adj', 'r2_nagelkerke'])
        ss.columns = ['Test Stat', 'P-value']
        self.sumstats = ss
    
    def predict(self):
        beta = self.params[-self.X.shape[1]:]
        yhat = self.X.dot(beta)
        th = np.concatenate([np.array([-1e6]), self.params[:-self.X.shape[1]],
                             np.array([1e6])])
        yhat = pd.cut(yhat, th).codes.astype(float)
        return yhat
    
    
    
    
    
    