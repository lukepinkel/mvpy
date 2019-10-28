#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:36:38 2019

@author: lukepinkel
"""


import patsy #analysis:ignore

import numpy as np  #analysis:ignore
import scipy as sp  #analysis:ignore
import scipy.stats  #analysis:ignore
import pandas as pd

from ..utils import linalg_utils, statfunc_utils, base_utils, data_utils  #analysis:ignore
'''

X_anova = np.zeros((1000, 3))
X_anova[:, 0] = np.random.normal(size=(1000))
X_anova[:, 1] = np.random.randint(0, 5, size=(1000))
X_anova[:, 2] = np.random.randint(0, 2, size=(1000))
X_anova = pd.DataFrame(X_anova, columns=['x1', 'x2', 'x3'])
X_dummy = patsy.dmatrix("~x1+C(x2)+C(x3)", X_anova, return_type='dataframe')
b_anova = np.random.normal(size=(7, 1))
y = sp.stats.matrix_normal(X_dummy.dot(b_anova), rowcov=np.eye(1000), 
                           colcov=np.array([[0.5]])).rvs()

df_anova = X_anova.copy()
df_anova['y'] = y


Y_anova, X_anova = patsy.dmatrices("y~x1+C(x2)+C(x3)", df_anova, 
                       return_type='dataframe')

Sx = mv.vine_corr(20, 5)
X = sp.stats.multivariate_normal(np.zeros(20), Sx).rvs(size=1000)
b = np.random.normal(size=(20, 1))

y = sp.stats.matrix_normal(X.dot(b), rowcov=np.eye(1000), colcov=np.array([[0.5]])).rvs()

df = pd.DataFrame(np.hstack([X, y]), columns=['x%i'%i for i in range(1, 21)]+['y'])

Y, X = patsy.dmatrices("y~"+"+".join(['x%i'%i for i in range(1, 21)]), df, 
                       return_type='dataframe')

nu = 1000 - 21
s2 = .5
v = sp.stats.invgamma(a=0.5*nu).rvs(1000)*0.5*s2*nu


S = mv.vine_corr(30, 10)

Sxx = mv.vine_corr(30, 10)
Syy = mv.vine_corr(10, 10)

X = sp.stats.multivariate_normal(np.zeros(30), Sxx).rvs(size=(1000))
b = np.random.normal(0, 2, size=(30, 10))
Y = sp.stats.matrix_normal(X.dot(b), rowcov=np.eye(1000), colcov=Syy).rvs()

'''

def is_pd(x):
    if (type(x) is pd.DataFrame)|((type(x) is pd.Series)):
        return True
    else:
        return False
    
        

class GeneralLinearModel(object):
    '''
    A potential broad class to reorganize linear models, hopefully including
    Generalized linear models(one parameter EDF), Cumulative link models, 
    Negative Binomial models, along with W/G/OLS, with another function 
    for ANOVA like partitioning of sum squares.  Also, experimenting with
    why multirep tests were giving such bizarre results.
    
    '''
    def __init__(self, formula=None, data=None, X=None, Y=None):
        
        if ((X is None)&(Y is None)):
            Y_df, X_df = patsy.dmatrices(formula, data, 
                                         return_type='dataframe')
            
        elif formula is None:
            if not is_pd(X):
                columns=['x%i'%i for i in range(1, X.shape[1]+1)]
                X = pd.DataFrame(X, columns=columns)
            if not is_pd(Y):
                columns=['y%i'%i for i in range(1, Y.shape[1]+1)]
                Y = pd.DataFrame(Y, columns=columns)
            Y_df, X_df = Y, X
        
     
        self.X_df, self.Y_df = X_df, Y_df
        self.X, self.xcols, self.xix, self.x_is_pd = base_utils.check_type(X_df)
        self.Y, self.ycols, self.yix, self.y_is_pd = base_utils.check_type(Y_df)
        n, p = self.X.shape
        q = self.Y.shape[1]
        r = np.linalg.matrix_rank(self.X)
        self.n, self.p, self.q, self.r = n, p, q, r
        
    def to_params(self):
        raise NotImplementedError
        
    def loglike(self, params):
        raise NotImplementedError
    
    def from_params(self, params):
        raise NotImplementedError
    
    def _fit_closed(self):
        raise NotImplementedError
    
    def _fit_iterative(self):
        raise NotImplementedError
    
    def model_hypothesis_testing(self):
        raise NotImplementedError
    
    def _multi_tests(self):
        raise NotImplementedError
    
    def _uni_tests(self):
        raise NotImplementedError
        
    def parameter_inference(self):
        raise NotImplementedError
    
    def goodness_of_fit(self):
        raise NotImplementedError
    
    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    


class OLS(GeneralLinearModel):

    def __init__(self, formula=None, data=None, X=None, Y=None, cov=None):
        
        super().__init__(formula, data, X, Y)
        self.cov = np.eye(self.n)
        
    
    def to_params(self, beta, Sigma):
        s = linalg_utils.vech(Sigma)
        theta = np.concatenate([linalg_utils.vec(beta), s])
        return theta
    
    def from_params(self, theta):
        beta, s = theta[:self.p], theta[self.p:]
        Sigma = linalg_utils.invech(s)
        beta = linalg_utils.invec(beta, self.p, self.q)
        return beta, Sigma
    
    def loglike(self, theta):
        beta, Sigma = self.from_params(theta)
        yhat = self.X.dot(beta)
        ll = sp.stats.matrix_normal(yhat, rowcov=np.eye(self.n), 
                                    colcov=Sigma).logpdf(self.Y).sum()
        return -ll
    
    def _fit_closed(self):
        X, Y = self.X, self.Y
        Sxx = np.dot(X.T, X)
        Sxy = np.dot(X.T, Y)
        Syy = np.dot(Y.T, Y)
        
        self.G = np.linalg.pinv(Sxx)
        self.beta = self.G.dot(Sxy)
        self.Sigma = (Syy - Sxy.T.dot(self.G).dot(Sxy))/(self.n - self.p)
        self.theta = self.to_params(self.beta, self.Sigma)
        self.Sxx, self.Sxy, self.Syy = Sxx, Sxy, Syy
    
    def model_hypothesis_testing(self, C=None, U=None, T=None):
        self.sumstats_multivar, _, self.rho2 = self._multi_tests(C, U, T)
        self.sumstats_univar = self._uni_tests()
    
    def _uni_tests(self):
        
        dfe = self.n - self.r - 1
        dfr = self.r - 1
        dft = self.n - 1
        Se = self.Sigma * (self.n - self.p)
        Sh =  np.linalg.multi_dot([self.beta.T, self.Sxx, self.beta])
        sumstats_univar = pd.DataFrame(np.zeros((self.q, 5)),
                                       columns=['F', 'p', 'R2', 'AdjR2', 'AICC'])
        sumstats_univar['F'] = np.diag(Sh/dfr) / np.diag(Se/dfe)
        sumstats_univar['p'] = sp.stats.f.sf(abs(sumstats_univar['F']), 
                       dfr, dfe)
        sumstats_univar['R2'] = np.diag(Sh) / np.diag(Sh+Se)
        sumstats_univar['AdjR2'] = 1 - np.diag(Se/dfe) / np.diag((Sh+Se)/dft)
        k = (2*self.r*self.n)/(self.n-self.r-1)
        sumstats_univar['AICC'] = self.n*np.log(np.diag(Se)/self.n)+k
        return sumstats_univar
        
    def _multi_tests(self, C=None, U=None, T=None):
        if C is None:
            C = np.eye(self.p)
        if U is None:
            U = np.eye(self.q)
        if T is None:
            T = np.zeros((self.p, self.q))
        Sxx_inv, dfe = self.G,  self.n - self.r

        M = np.linalg.multi_dot([C, Sxx_inv, C.T])
        Sigma_star = np.linalg.multi_dot([U.T, self.Sigma, U]) * (self.n - self.p)
        Minv = np.linalg.pinv(M)
        TA = np.linalg.multi_dot([C, self.beta, U])
        Td = TA - T
        Se = dfe * Sigma_star
        Sh = np.linalg.multi_dot([Td.T, Minv, Td])
        a, b, dfe = self.p, self.q, self.n - self.r
        a2, b2 = a**2, b**2
        if a2*b2 <= 4:
            g = 1
        else:
            g = (a2*b2-4) / (a2 + b2 - 5)
        
        rho2, _ = np.linalg.eig(Sh.dot(np.linalg.inv(Sh+Se/dfe)))
        s = np.min([a, b])
        tst_hlt = np.sum(rho2/(1-rho2))
        tst_pbt = np.sum(rho2)
        tst_wlk = np.product(1-rho2)
        
        eta_hlt = (tst_hlt/s) / (1 + tst_hlt/s)
        eta_pbt = tst_pbt / s
        eta_wlk = 1 - np.power(tst_wlk, (1/g))
        
        test_stats = np.vstack([tst_hlt, tst_pbt, tst_wlk]).T
        effect_sizes = np.vstack([eta_hlt, eta_pbt, eta_wlk]).T
        test_stats = pd.DataFrame(test_stats, columns=['HLT', 'PBT', 'WLK'])
        effect_sizes = pd.DataFrame(effect_sizes, columns=['HLT', 'PBT', 'WLK'])
        
        df_hlt1 = a * b
        df_wlk1 = a * b

        df_pbt1 = s * (dfe + s - b) * (dfe + a + 2) * (dfe + a - 1)
        df_pbt1 /= (dfe * (dfe + a - b))
        df_pbt1 -= 2
        df_pbt1 *= (a * b) / (s * (dfe + a))

        df_hlt2 = (dfe**2 - dfe * (2 * b + 3) + b * (b + 3)) * (a * b + 2)
        df_hlt2 /= (dfe * (a + b + 1) - (a + 2 * b + b2 - 1))
        df_hlt2 += 4

        df_pbt2 = s * (dfe + s - b) * (dfe + a + 2) * (dfe + a - 1)
        df_pbt2 /= dfe * (dfe + a - b)
        df_pbt2 -= 2
        df_pbt2 *= (dfe + s - b) / (dfe + a)

        df_wlk2 = g * (dfe - (b - a + 1) / 2) - (a * b - 2) / 2
        df1 = np.array([df_hlt1, df_pbt1, df_wlk1])
        df2 = np.array([df_hlt2, df_pbt2, df_wlk2])
        f_values = (effect_sizes / df1) / ((1 - effect_sizes) / df2)
        p_values = sp.stats.f.sf(f_values, df1, df2)
        p_values = pd.DataFrame(p_values, columns=effect_sizes.columns)
        df1 = pd.DataFrame(df1, index=effect_sizes.columns).T
        df2 = pd.DataFrame(df2, index=effect_sizes.columns).T
        
        sumstats = pd.concat([test_stats, effect_sizes, f_values, df1, df2, 
                              p_values])
        sumstats.index = ['Test Stat', 'Eta', 'F-values', 'df1', 'df2', 'P-values']
        sumstats = sumstats.T
        return sumstats, effect_sizes, rho2
    
    def parameter_inference(self):
        self.Vbeta = np.outer(np.diag(self.G), np.diag(self.Sigma))
        Sinv = np.linalg.pinv(self.Sigma)
        self.Vsigma = linalg_utils.dmat(self.q).T.dot(np.diag(0.5 * np.kron(Sinv,
                                       Sinv)))
        self.SEb = linalg_utils.vecc(np.sqrt(self.Vbeta))
        self.SEs = np.sqrt(self.Vsigma)
        self.SEt = np.concatenate([self.SEb, linalg_utils._check_2d(self.SEs)])
    
    def fit(self):
        self._fit_closed()
        self.model_hypothesis_testing()
        self.parameter_inference()
        ylabels = list(zip(self.ycols[np.tril_indices(self.q)[0]], 
                           self.ycols[np.tril_indices(self.q)[1]]))
        xlabels = ['%s ~ %s'%(x, y) for y in self.ycols for x in self.xcols]
        self.t_values = linalg_utils._check_2d(self.theta)  / self.SEt
        self.res = pd.DataFrame(np.hstack([linalg_utils._check_2d(self.theta),
                                           self.SEt, self.t_values]))
        self.res.index = xlabels + ylabels
        self.res.columns = ['param', 'SE', 't_value']
        self.res['p_value'] = sp.stats.t.sf(abs(self.res['t_value']), 
                self.n-self.p)*2.0
                

    
'''       
ols_model = OLS("y~"+"+".join(['x%i'%i for i in range(1, 21)]), df)
ols_model.fit()
ols_model._fit_closed()     
ols_model.loglike(ols_model.theta)        
ols_model.model_hypothesis_testing()     
ols_model.parameter_inference()     
ols_model.theta[:, None] / ols_model.SEt
        
ols_model = OLS(X=X, Y=Y)
ols_model.fit()

'''

