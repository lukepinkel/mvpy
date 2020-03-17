#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 20:17:57 2020

@author: lukepinkel
"""
import autograd # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd  # analysis:ignore
import matplotlib as mpl  # analysis:ignore
import autograd.numpy as np # analysis:ignore


        
    
class ODESystem(object):
    
    def __init__(self, func, y0, params, t):
        
        if type(params) in [int, float]:
            params = np.array([params])
            
        if type(y0) in [int, float]:
            y0 = np.array([y0])
        
      
        self.y0 = y0
        self.t = t
        self.func = func
        self.params = params
        self.p, self.q = len(y0), len(params)
        
        self.u0 = np.concatenate([self.y0, np.zeros(self.q*self.p)])
        self.dfdy = autograd.jacobian(func, argnum=0)
        self.dfdp = autograd.jacobian(func, argnum=2)
        self.partition = len(y0)
     
    def _grad(self, u, t, params):
        y, z = u[:self.partition], u[self.partition:].reshape((self.p, self.q))
        dy = self.func(y, t, params)
        dz = np.dot(self.dfdy(y, t, params), z) + self.dfdp(y, t, params)
        du = np.concatenate([dy, dz.reshape(-1)])
        return du
    
    def _split_u(self, u):
        y = u[:, :self.p]
        z = u[:, self.p:].reshape((u.shape[0], self.p, self.q))
        return y, z
    
    def _jvp(self, u):
        y, z = self._split_u(u)
        z = z.reshape((z.shape[0]*self.p, self.q))
        y = y.reshape((y.shape[0]*self.p,))
        g = np.dot(z.T, y)
        return g
        
    def integrate(self, params=None, sensitivities=True):
        if params is None:
            params = self.params
        if sensitivities:
            y = sp.integrate.odeint(self._grad, self.u0, self.t, args=(params,))
        else:
            y = sp.integrate.odeint(self.func, self.y0, self.t, args=(params,))
        return y
 
    
class ODEModel(object):
    
    def __init__(self, odesys, y_obs, weights=None):
        if weights is None:
            weights = np.eye(y_obs.shape[1])
        
        self.odesys = odesys
        self.y_obs = y_obs
        self.jac = autograd.jacobian(self.sse, argnum=1)
        self.weights = weights
        
        
    @classmethod
    def sse(cls, yhat, y):
        r = yhat - y
        return np.dot(r.T, r) / 2
    
    def loglike(self, params, ret_jac=True):
        if ret_jac:
            u = self.odesys.integrate(params)
            g = self._score(params, u)
            y, _ = self.odesys._split_u(u)
            ll = np.trace(self.sse(y, self.y_obs).dot(self.weights))
            return ll, g           
        else:
            u = self.odesys.integrate(params)
            y, _ = self.odesys._split_u(u)
            ll = np.trace(self.sse(y, self.y_obs).dot(self.weights))
            return ll
         
    def _score(self, params, u):
        y, z = self.odesys._split_u(u)
        dydp = z.reshape((z.shape[0]*self.odesys.p, self.odesys.q))
        dldy = self.jac(y.reshape(-1), self.y_obs.reshape(-1))  
        g = np.dot(dldy.T, dydp)
        return -g
    
    def score(self, params):
        u = self.odesys.integrate(params)
        g = self._score(params, u)
        return g
'''    
def fhn(y, t, params):
    y1, y2 = y
    a, b, c = params
    
    dy1 = (y1 - y1**3 / 3 + y2) * c
    dy2 =-(y1 - a + b * y2) / c
    return np.array([dy1, dy2])
 

params = np.array([0.2, 0.2, 3.0])
y0 = np.array([1.0, 0.0])

odesys = ODESystem(fhn, y0, params, np.linspace(0, 20, 200))

y_obs = odesys.integrate(np.array([0.2, 0.2, 3.0]), sensitivities=False)



odemod = ODEModel(odesys, y_obs)

odemod.score(np.array([0.2, 0.2, 3.0]))
odemod.score(np.array([0.1, 0.3, 3.0]))

odemod.loglike(np.array([0.2, 0.2, 3.0]))

odemod.loglike(np.array([0.2, 0.2, 3.0]), False)

odemod.loglike(np.array([0.1, 0.3, 3.0]), True)

odemod.loglike(np.array([0.1, 0.1, 3.0]), True)
odemod.loglike(np.array([0.3, 0.3, 3.0]), True)
'''



