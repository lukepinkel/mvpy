#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:25:20 2019

@author: lukepinkel
"""
import patsy

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

from ..utils import linalg_utils, base_utils, statfunc_utils


class LM:

    def __init__(self, formula, data):
        y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
        self.X, self.y, self.data = X, y, data
        self.sumstats = self.lmss()
        self.gram = np.linalg.pinv(X.T.dot(X))
        yp = linalg_utils._check_np(y)
        self.coefs = linalg_utils._check_np(self.gram.dot(X.T)).dot(yp)
        self.yhat = linalg_utils._check_np(X).dot(self.coefs)
        self.sse = np.sum((linalg_utils._check_np(y) - self.yhat)**2,
                          axis=0)
        self.error_var = self.sse / (X.shape[0] - X.shape[1] - 1)
        self.coefs_se = np.sqrt(np.diag(self.gram*self.error_var))
        self.ll = self.loglike(self.coefs, self.error_var)
        beta = np.concatenate([linalg_utils._check_2d(self.coefs),
                              linalg_utils._check_2d(self.coefs_se)],
                              axis=1)
        self.res = pd.DataFrame(beta, index=self.X.columns,
                                columns=['beta', 'SE'])
        self.res['t'] = self.res['beta'] / self.res['SE']
        self.res['p'] = sp.stats.t.sf(abs(self.res['t']),
                                      X.shape[0]-X.shape[1])*2.0

    def lmss(self):
        X, Y, data = self.X, self.y, self.data
        
        di = X.design_info
        di = X.design_info
        term_names = di.term_names
        column_names = di.subset(term_names).column_names
        X = X[column_names]
        G = pd.DataFrame(np.linalg.inv(X.T.dot(X)), 
                         index=X.columns, columns=X.columns)
        B = G.dot(X.T).dot(Y)
        
        SSE = np.sum((Y - X.dot(B))**2)
        SST = np.sum((Y - Y.mean())**2)
        SSH = B.T.dot(X.T).dot(Y-Y.mean())
        SSI = B.T.dot(X.T).dot(Y) - SSH
        
        sumsq = {}
        meansq = {}
        dfs = {}
        for term in term_names:
            if term!='Intercept':
                Xk = patsy.dmatrix('~'+term, data=data, return_type='dataframe')
                SSHk = np.linalg.lstsq(Xk, Y)[0].T.dot(Xk.T).dot(Y) - SSI
                dfk = Xk.shape[1] - 1
                if term.find(':')!=-1:
                    for x in term_names:
                        if (x!=term)&(x in term):
                            SSHk -= sumsq[x]
                            dfk -= dfs[x]
                sumsq[term] = linalg_utils._check_0d(linalg_utils._check_np(SSHk))
                dfs[term] = linalg_utils._check_0d(linalg_utils._check_np(dfk))
                meansq[term] = linalg_utils._check_0d(linalg_utils._check_np(SSHk/dfk))
                
        anova = pd.DataFrame([sumsq, dfs, meansq], index=['SSQ', 'df', 'MSQ']).T
        rss = linalg_utils._check_0d(linalg_utils._check_np(SSE))
        dfr = X.shape[0] - linalg_utils._check_0d(linalg_utils._check_np(anova['df'].sum()))
        anova['F'] = anova['MSQ'] / (rss/dfr)
        anova['P'] = sp.stats.f.sf(anova['F'], anova['df'], dfr-1)
        
        
        anr = pd.DataFrame([[rss, dfr, rss/dfr, '-', '-']],  columns=anova.columns,
                           index=['Residual'])
        anova = pd.concat([anova, anr])
        anova['r2'] = anova['SSQ'] / linalg_utils._check_0d(linalg_utils._check_np(SST))
        return anova

    def minimum_ols(self, X, y):
        n, p = X.shape
        dfe = n - p - 1.0
        dft = n
        dfr = p - 1.0

        sst = np.var(y, axis=0)*y.shape[0]

        G = np.linalg.pinv(X.T.dot(X))
        beta = G.dot(X.T.dot(y))
        yhat = linalg_utils._check_np(X.dot(beta))

        ssr = np.sum((yhat - linalg_utils._check_np(np.mean(y, axis=0)))**2,
                     axis=0)
        sse = np.sum((yhat - linalg_utils._check_np(y))**2, axis=0)

        res = [sst, ssr, sse, sst/dft, ssr/dfr, sse/dfe, ssr/sst,
               1.0 - (sse/dfe)/(sst/dft)]
        res = [linalg_utils._check_np(linalg_utils._check_1d(x)) for x in res]
        return res

    def loglike(self, coefs, error_var):
        yhat = linalg_utils._check_np(self.X).dot(coefs)
        error = linalg_utils._check_np(self.y) - yhat
        ll = sp.stats.norm.logpdf(error, loc=0, scale=error_var**.5).sum()
        return ll

    def _htest(self, X):
        anv = np.concatenate(self.minimum_ols(X, (self.y - self.yhat)**2))
        anova_table = pd.DataFrame(anv, index=['sst', 'ssr', 'sse', 'mst',
                                               'msr', 'mse', 'r2', 'r2_adj']).T
        anova_table['F'] = anova_table.eval('msr/mse')
        chi2 = linalg_utils._check_np(anova_table['r2'] * self.X.shape[0])
        chi2 = linalg_utils._check_0d(chi2)
        chi2p = linalg_utils._check_np(sp.stats.chi2.sf(chi2,
                                                        self.X.shape[1]-1))
        chi2p = linalg_utils._check_0d(chi2p)
        f = linalg_utils._check_0d(linalg_utils._check_np(anova_table['F']))
        fp = sp.stats.f.sf(f, self.X.shape[0]-self.X.shape[1],
                           self.X.shape[1]-1)
        htest_res = [[chi2, chi2p], [f, fp]]
        htest_res = pd.DataFrame(htest_res, index=['chi2', 'F'],
                                 columns=['Test Value', 'P Value'])
        return htest_res

    def _whitetest(self):
        X = linalg_utils._check_np(self.X)
        Xprods = X[:, np.tril_indices(X.shape[1])[0]]
        Xprods *= Xprods[:, np.tril_indices(X.shape[1])[1]]
        return self._htest(Xprods)

    def _breuschpagan(self):
        return self._htest(self.X)

    def heteroskedasticity_test(self):
        res = pd.concat([self._whitetest(),
                         self._breuschpagan()])
        res.index = pd.MultiIndex.from_product([['White Test',
                                                 'Breusch-Pagan'],
                                               res.index[:2].tolist()])
        self.heteroskedasticity_res = res

    def predict(self, X=None, ci=None, pi=None):
        if X is None:
            X = self.X
        X, xcols, xix, x_is_pd = base_utils.check_type(X)
        yhat = X.dot(self.coefs)
        yhat = pd.DataFrame(yhat, index=xix, columns=['yhat'])
        if ci is not None:
            nme = int(ci)
            ci = (100.0 - ci) / 200.0
            cip, cim = 1.0 - ci, ci
            zplus, zminus = sp.stats.norm.ppf(cip), sp.stats.norm.ppf(cim)
            s2 = np.sqrt(linalg_utils._check_0d(self.error_var))
            yhat['CI%i-' % nme] = yhat['yhat'] + zminus * s2
            yhat['CI%i+' % nme] = yhat['yhat'] + zplus * s2
        if pi is not None:
            nme = int(pi)
            pi = (100.0 - pi) / 200.0
            pip, pim = 1.0 - pi, pi
            zplus, zminus = sp.stats.norm.ppf(pip), sp.stats.norm.ppf(pim)
            error_var = linalg_utils._check_0d(self.error_var)
            s2 = np.diag(X.dot(self.gram).dot(X.T)) * error_var
            yhat['PI%i-' % nme] = yhat['yhat'] + s2 * zminus
            yhat['PI%i+' % nme] = yhat['yhat'] + s2 * zplus
        return yhat


class MassUnivariate:

    def __init__(self, X, Y):
        R = base_utils.corr(X, Y)
        Z = pd.concat([X, Y], axis=1)
        u, V = np.linalg.eigh(base_utils.corr(Z))
        u = u[::-1]
        eigvar = np.sum((u - 1.0)**2) / (len(u) - 1.0)

        m_eff1 = 1 + (len(u) - 1.0) * (1 - eigvar / len(u))
        m_eff2 = np.sum(((u > 1.0) * 1.0 + (u - np.floor(u)))[u > 0])
        m_eff3 = np.sum((u.cumsum() / u.sum()) < 0.99)

        N = base_utils.valid_overlap(X, Y)
        df = N - 2
        t_values = R * np.sqrt(df / np.maximum((1 - R**2), 1e-16))
        p_values = pd.DataFrame(sp.stats.t.sf(abs(t_values), df)*2.0,
                                index=X.columns, columns=Y.columns)
        minus_logp = pd.DataFrame(-sp.stats.norm.logsf(abs(t_values))/2.0,
                                  index=X.columns, columns=Y.columns)
        res = pd.concat([R.stack(), t_values.stack(), p_values.stack(),
                         minus_logp.stack()], axis=1)
        res.columns = ['correlation', 't_value', 'p_value', 'minus_logp']
        res['p_fdr'] = statfunc_utils.fdr_bh(res['p_value'].values)
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
        self.t_values, self.p_values = t_values, p_values
        self.minus_log = minus_logp

        self.res = res
   

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
    
    def gradient(self, params):
        raise NotImplementedError
    
    def hessian(self, params):
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
        beta, s = theta[:self.p*self.q], theta[self.p*self.q:]
        Sigma = linalg_utils.invech(s)
        beta = linalg_utils.invec(beta, self.p, self.q)
        return beta, Sigma
    
    def loglike(self, theta):
        beta, Sigma = self.from_params(theta)
        yhat = self.X.dot(beta)
        ll = sp.stats.matrix_normal(yhat, rowcov=np.eye(self.n), 
                                    colcov=Sigma).logpdf(self.Y).sum()
        return -ll
    
    def gradient(self, theta):
        beta, Sigma = self.from_params(theta)
        X, Y = self.X, self.Y
        g = linalg_utils.vec(np.dot(X.T, Y-X.dot(beta)))
        return g
    
    def hessian(self, theta):
        beta, Sigma = self.from_params(theta)
        X, Y = self.X, self.Y
        E = Y-X.dot(beta)
        Sinv = np.linalg.pinv(Sigma)
        Hbb = np.kron(Sinv, X.T.dot(X))
        Hbs = np.kron(Sinv, np.linalg.multi_dot([X.T, E, Sinv]))
        Hss = np.kron(Sinv, Sinv)/2.0
        H = np.block([[Hbb, Hbs], [Hbs.T, Hss]])
        return H
        
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
            g = np.sqrt((a2*b2-4) / (a2 + b2 - 5))
        
        rho2, _ = np.linalg.eig(Sh.dot(np.linalg.inv(Sh+Se/dfe)))
        s = np.min([a, b])
        tst_hlt = np.sum(rho2/(1-rho2))
        tst_pbt = np.sum(rho2)
        tst_wlk = np.product(1-rho2)
        tst_rlr = np.max(rho2/(1-rho2))
        
        eta_hlt = (tst_hlt/s) / (1 + tst_hlt/s)
        eta_pbt = tst_pbt / s
        eta_wlk = 1 - np.power(tst_wlk, (1/g))
        eta_rlr = np.max(rho2)
        
        test_stats = np.vstack([tst_hlt, tst_pbt, tst_wlk,]).T
        effect_sizes = np.vstack([eta_hlt, eta_pbt, eta_wlk]).T
        test_stats = pd.DataFrame(test_stats, columns=['HLT', 'PBT', 'WLK'])
        effect_sizes = pd.DataFrame(effect_sizes, columns=['HLT', 'PBT', 'WLK'])
        
        df_hlt1 = a * b
        df_wlk1 = a * b
        df_rlr1 = np.max([a, b])

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
        df_rlr2 = dfe - np.max([a, b]) + b
        df1 = np.array([df_hlt1, df_pbt1, df_wlk1])
        df2 = np.array([df_hlt2, df_pbt2, df_wlk2])
        f_values = (effect_sizes / df1) / ((1 - effect_sizes) / df2)
        p_values = sp.stats.f.sf(f_values, df1, df2)
        p_values = pd.DataFrame(p_values, columns=effect_sizes.columns)
        df1 = pd.DataFrame(df1, index=effect_sizes.columns).T
        df2 = pd.DataFrame(df2, index=effect_sizes.columns).T
        f_rlr = tst_rlr * (df_rlr2 / df_rlr1)
        p_rlr = sp.stats.f.sf(f_rlr, df_rlr1, df_rlr2)
        rlr = pd.DataFrame([tst_rlr, eta_rlr, f_rlr, df_rlr1, df_rlr2, p_rlr])
        rlr.index = ['Test Stat', 'Eta', 'F-values', 'df1', 'df2', 'P-values']
        rlr.columns = ['RLR']
        rlr = rlr.T
        sumstats = pd.concat([test_stats, effect_sizes, f_values, df1, df2, 
                              p_values])
        sumstats.index = ['Test Stat', 'Eta', 'F-values', 'df1', 'df2', 'P-values']
        sumstats = sumstats.T
        sumstats = pd.concat([sumstats, rlr])
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
                
    