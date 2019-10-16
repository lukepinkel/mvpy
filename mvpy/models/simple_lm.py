#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:25:20 2019

@author: lukepinkel
"""
import patsy
import collections
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

from ..utils import linalg_utils, base_utils, statfunc_utils


class LM:

    def __init__(self, formula, data):
        y, X = patsy.dmatrices(formula, data=data, return_type='dataframe')
        self.X, self.y = X, y
        self.sumstats = self.lmss(X, y)
        self.gram = np.linalg.pinv(X.T.dot(X))
        yp = linalg_utils._check_np(y)
        self.coefs = linalg_utils._check_np(self.gram.dot(X.T)).dot(yp)
        self.yhat = linalg_utils._check_np(X).dot(self.coefs)
        self.error_var = np.sum((linalg_utils._check_np(y) - self.yhat)**2,
                                axis=0) / (X.shape[0] - X.shape[1] - 1)
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

    def lmss(self, X, y):
        di = X.design_info
        # if 'Intercept' in X.columns.tolist():
        #    dnames = di.term_names[1:]
        # else:
        #    dnames = di.term_names
        Xmats = [X.loc[:, di.subset(x).column_names]
                 for x in di.term_names[1:]]
        # Xmats = [X.iloc[:, di.slice(x)] for x in dnames]
        Xmats = collections.OrderedDict(zip(di.term_names[1:], Xmats))

        anv = collections.OrderedDict()
        for key in Xmats.keys():
            anv[key] = np.concatenate(self.minimum_ols(Xmats[key], y))
        anv['Total'] = np.concatenate(self.minimum_ols(self.X, self.y))
        anova_table = pd.DataFrame(anv, index=['sst', 'ssr', 'sse', 'mst',
                                               'msr', 'mse', 'r2', 'r2_adj']).T

        anova_table['F'] = anova_table.eval('msr/mse')
        return anova_table

    def minimum_ols(self, X, y):
        n, p = X.shape
        dfe = n - p - 1.0
        dft = n - 1.0
        dfr = p

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
        n = self.X.shape[0]
        k = n/2.0 * np.log(2*np.pi)
        ldt = n/2.0*np.log(error_var)
        dev = 0.5
        ll = k + ldt + dev
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



class LinearModel:

    def __init__(self, X, Y, C=None, U=None, T0=None):

        X, xcols, xix, x_is_pd = base_utils.check_type(X)
        Y, ycols, yix, y_is_pd = base_utils.check_type(Y)
        Sxx, Sxy, Syy = np.dot(X.T, X), np.dot(X.T, Y), np.dot(Y.T, Y)
        Sxx_inv = np.linalg.pinv(Sxx)
        n, p, _, q = *X.shape, *Y.shape
        r = np.linalg.matrix_rank(X)
        if C is None:
            C = np.eye(p)
        if U is None:
            U = np.eye(q)
        if T0 is None:
            T0 = np.zeros((p, q))

        self.X, self.xcols, self.xix, self.x_is_pd = X, xcols, xix, x_is_pd
        self.Y, self.ycols, self.yix, self.y_is_pd = Y, ycols, yix, y_is_pd
        self.Sxx, self.Sxx_inv, self.Sxy, self.Syy = Sxx, Sxx_inv, Sxy, Syy
        self.n_obs, self.p_vars, self.q_vars, self.xrank = n, p, q, r
        self.dfe = n - r
        self.C, self.U, self.T0 = C, U, T0

    def _get_coefs(self):
        Sxx_inv, Sxy, Syy, dfe = self.Sxx_inv, self.Sxy, self.Syy, self.dfe
        B = Sxx_inv.dot(Sxy)
        Sigma = (Syy - np.dot(Sxy.T, B)) / dfe
        return B, Sigma

    def _get_model_mats(self, C=None, U=None, T0=None, B=None, Sigma=None):
        if C is None:
            C = self.C
        if U is None:
            U = self.U
        if T0 is None:
            T0 = self.T0

        if B is None:
            if hasattr(self, 'B'):
                B, Sigma = self.B, self.Sigma
            else:
                B, Sigma = self._get_coefs()
        Sxx_inv, dfe = self.Sxx_inv, self.dfe

        M = np.linalg.multi_dot([C, Sxx_inv, C.T])
        Sigma_star = np.linalg.multi_dot([U.T, Sigma, U])
        Minv = np.linalg.pinv(M)
        TA = np.linalg.multi_dot([C, B, U])
        Td = TA - T0
        Se = dfe * Sigma_star
        Sh = np.linalg.multi_dot([Td.T, Minv, Td])
        Omega = Sh.dot(np.linalg.pinv(Sigma_star))
        return Se, Sh, Omega

    def _test_stats(self, Se=None, Sh=None, Omega=None):
        if Se is None:
            Se, Sh, Omega = self._get_model_mats()
        a, b, dfe = self.p_vars, self.q_vars, self.dfe
        a2, b2 = a**2, b**2
        if a2*b2 <= 4:
            g = 1
        else:
            g = (a2*b2-4) / (a2 + b2 - 5)
        w, _ = np.linalg.eig(Omega)
        s = np.min([a, b])
        rho2 = w / (w + self.n_obs)

        hlt = np.sum(rho2/(1-rho2))
        pbt = np.sum(rho2)
        wlk = np.product(1-rho2)
        test_stats = np.array([hlt, pbt, wlk])

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
        u = hlt / s
        effects = np.array([u / (1 + u), pbt / s, 1 - np.power(wlk, 1.0/g)])
        df1 = np.array([df_hlt1, df_wlk1, df_pbt1])
        df2 = np.array([df_hlt2, df_wlk2, df_pbt2])
        sumstats = np.vstack([test_stats, effects, df1, df2]).T
        sumstats = pd.DataFrame(sumstats, index=['Hotelling Trace',
                                                 'Pillai Trace',
                                                 'Wilks Lambda'])
        sumstats.columns = ['Test Stat', 'Effect Size', 'df1', 'df2']
        eta = sumstats['Effect Size']
        df1, df2 = sumstats['df1'], sumstats['df2']
        sumstats['F-value'] = (eta / df1) / ((1 - eta) / df2)
        sumstats['P-value'] = sp.stats.f.sf(sumstats['F-value'],
                                            sumstats['df1'],
                                            sumstats['df2'])
        return sumstats
