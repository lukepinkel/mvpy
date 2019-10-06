#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:25:20 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy as sp

from collections import OrderedDict
from patsy import dmatrices
from numpy.linalg import pinv
from scipy.stats import t as t_dist

from ..utils.linalg_utils import einv, _check_np, _check_1d, _check_2d
from ..utils.base_utils import corr, valid_overlap
from ..utils.statfunc_utils import fdr_bh
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
    
class MassUnivariate:
    
    def __init__(self, X, Y):
        
      R = corr(X, Y)
      Z = pd.concat([X, Y], axis=1)
      u, V = np.linalg.eigh(corr(Z))
      u = u[::-1]
      eigvar = np.sum((u - 1.0)**2) / (len(u) - 1.0)
      
      m_eff1 = 1 + (len(u) - 1.0) * (1 - eigvar / len(u))
      m_eff2 = np.sum(((u>1.0)*1.0 + (u - np.floor(u)))[u>0])
      m_eff3 = np.sum((u.cumsum() / u.sum())<0.99)
      
      N = valid_overlap(X, Y)
      df = N - 2
      t_values = R * np.sqrt(df / np.maximum((1 - R**2), 1e-16))
      p_values = pd.DataFrame(sp.stats.t.sf(abs(t_values), df)*2.0, index=X.columns,
                              columns=Y.columns)
      minus_logp = pd.DataFrame(-sp.stats.norm.logsf(abs(t_values))/2.0, index=X.columns,
                                columns=Y.columns)
      res = pd.concat([R.stack(), t_values.stack(), p_values.stack(),
                       minus_logp.stack()], axis=1)
      res.columns = ['correlation', 't_value', 'p_value', 'minus_logp']
      res['p_fdr'] = fdr_bh(res['p_value'].values)
      res['p_meff1_sidak'] = 1.0 - (1.0 - res['p_value'])**m_eff1
      res['p_meff2_sidak'] = 1.0 - (1.0 - res['p_value'])**m_eff2
      res['p_meff3_sidak'] = 1.0 - (1.0 - res['p_value'])**m_eff3
      
      res['p_meff1_bf'] = np.minimum(res['p_value']*m_eff1, 0.5-1e-16)
      res['p_meff2_bf'] = np.minimum(res['p_value']*m_eff2, 0.5-1e-16)
      res['p_meff3_bf'] = np.minimum(res['p_value']*m_eff3, 0.5-1e-16)
      
      self.m_eff1, self.m_eff2, self.m_eff3 = m_eff1, m_eff2, m_eff3
      self.X, self.Y = X, Y
      self.R, self.N, self.df = R, N, df
      self.eigvals, self.eigvecs = u, V
      self.t_values, self.p_values, self.minus_logp = t_values, p_values, minus_logp
      self.res = res
     
        



    
      
      
      
      
        
        
        
        
        
        
        
        
        
        

    
    