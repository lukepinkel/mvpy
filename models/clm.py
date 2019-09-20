#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:41:56 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd

from numpy import exp, ones, dot, diag
from patsy import dmatrices
from scipy.optimize import minimize
from ..utils.linalg_utils import einv, _check_1d, mdot
from ..utils.base_utils import check_type
from ..utils.statfunc_utils import norm_qtf

class CLM:
    
    def __init__(self, frm, data, X=None, Y=None):
        Y, X = dmatrices(frm, data, return_type='dataframe')
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_p = check_type(Y)
        self.Y = _check_1d(self.Y)
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
        X, Y, W = self.X, self.Y, self.W
        params = _check_1d(params)
        A1, A2 = Y[:, :-1], Y[:, 1:]
        o1, o2 = Y[:, -1]*10e1, Y[:, 0]*-10e5
        B1, B2 = np.block([A1, -X]), np.block([A2, -X])
        Nu_1, Nu_2 = dot(B1, params)+o1, dot(B2, params)+o2
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        Pi = Gamma_1 - Gamma_2
        LL = np.sum(W * np.log(Pi))
        return -LL
    
    def gradient(self, params):
        X, Y, W = self.X, self.Y, self.W
        A1, A2 = Y[:, :-1], Y[:, 1:]
        B1, B2 = np.block([A1, -X]), np.block([A2, -X])
        o1, o2 = Y[:, -1]*10e1, Y[:, 0]*-10e5
        Nu_1, Nu_2 = dot(B1, params)+o1, dot(B2, params)+o2
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        Phi_11, Phi_12 = diag(Phi_11), diag(Phi_12)
        dPi = dot(B1.T, Phi_11) - dot(B2.T, Phi_12)
        
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        
        g = -dot(dPi, W / Pi)
        return g
    
    def hessian(self, params):
        X, Y, W = self.X, self.Y, self.W
        A1, A2 = Y[:, :-1], Y[:, 1:]
        B1, B2 = np.block([A1, -X]), np.block([A2, -X])
        o1, o2 = Y[:, -1]*10e1, Y[:, 0]*-10e5
        Nu_1, Nu_2 = dot(B1, params)+o1, dot(B2, params)+o2
        Phi_21, Phi_22 = self.d2logit(Nu_1),  self.d2logit(Nu_2)
        #Phi_21[Y[:, 0].astype(bool)], Phi_22[Y[:, -1].astype(bool)] = 1.0, 0.0
    
        Phi_11, Phi_12 = self.dlogit(Nu_1), self.dlogit(Nu_2)
        #Phi_11[Y[:, 0].astype(bool)], Phi_12[Y[:, -1].astype(bool)] = 1.0, 0.0
        Phi_11, Phi_12 = diag(Phi_11), diag(Phi_12)    
        Phi_21, Phi_22 = diag(Phi_21), diag(Phi_22)
        Gamma_1, Gamma_2 = self.inv_logit(Nu_1), self.inv_logit(Nu_2)
        #Gamma_1[Y[:, 0].astype(bool)] = 1.0
        #Gamma_2[Y[:, -1].astype(bool)] = 0.0
        Pi = Gamma_1 - Gamma_2
        Phi3 = diag(W / Pi**2)
        dPi = dot(B1.T, Phi_11) - dot(B2.T, Phi_12)
        H=mdot([B1.T, Phi_21, B1])-mdot([B2.T, Phi_22, B2])-mdot([dPi, Phi3, dPi.T])
        return -H
    
    def fit(self, verbose=2, optim='double', lqp_kws={}, trust_kws={}, 
            trust_opts={}):
        
        theta = norm_qtf(np.sum(self.Y, axis=0).cumsum()[:-1]/np.sum(self.Y))
        beta = ones(self.X.shape[1])
        params = np.concatenate([theta, beta], axis=0)
        self.theta_init = theta
        
        #A little overkill
        res = minimize(self.loglike, params, constraints=self.constraints,
                       method='SLSQP', **lqp_kws)
        
        if optim.lower() in ['double', 'dual', '2']:
            
            options = {'verbose':verbose}
            for x in trust_opts.keys():
                options[x] = trust_opts[x]
                
            res = minimize(self.loglike, res.x, constraints=self.constraints,
                           jac=self.gradient, hess=self.hessian,
                           method='trust-constr', options=options,
                           **trust_kws)
            
        self.params = res.x
        self.optimizer = res
        self.H = self.hessian(self.params)
        self.Vcov = einv(self.H)
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
        self.theta = self.params[:len(theta)]
        self.beta = self.params[len(theta):]
        
