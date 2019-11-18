#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:46:32 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.optimize import minimize
from statsmodels.iolib.table import SimpleTable

from ..utils import statfunc_utils, linalg_utils
from ..utils.base_utils import corr, cov, check_type


class EFA:

    def __init__(self, data, method='corr',
                 n_iters=1000):
        if method == 'corr':
            S = corr(data)
            if np.min(np.linalg.eig(S)[0]) < 0:
                S = linalg_utils.near_psd(S)
        elif method == 'cov':
            S = cov(data)
        u, V = linalg_utils.sorted_eig(S)
        self.n_iters = n_iters
        self.eig_expvar = u / u.sum()
        self.data = data
        self.S = S
        self.eigvals, self.eigvecs = u, V
        self.n, self.p = data.shape
        self.data, self.cols, self.ix, self.is_pd = check_type(data)

        self.MSA = statfunc_utils.msa(S)
        self.lndetS = np.linalg.slogdet(S)[1]
        self.Sinv = np.linalg.pinv(S)
    
    def implied_cov(A, Psi=None, psi=None):
        if Psi is None:
            Psi = np.diag(linalg_utils._check_1d(psi))
        Sigma = np.dot(A, A.T)+Psi
        return Sigma
        
    def loglike(self, Sigma):
        '''
        Computes the log likelihood of a factor analysis model

        Parameters:
            R: Correlation matrix
            A: Loadings matrix
            psi: uniqueness, the complement of communalities

        Returns:
            ll: log likelihood without the -n/2 term
        '''
        lnS = np.linalg.slogdet(Sigma)[1]
        ll = np.trace(np.dot(np.linalg.pinv(Sigma), self.S)) + lnS
        ll-= self.lndetS
        ll-= self.Sigma.shape[0]
        return ll

    def _pfa(self, n_comps, n_iters=100, tol=1e-6,
                             vocal=False):
        '''
        Principal axis factor analysis

        Parameters:
            R: Correlation matrix
            A: Initial loadings matrix, normally principal components
            psi: Initial estimate of uniqueness
            ncomps: Number of factors
            n_iters: Number of iterations
            tol: Tolerance before convergance
            vocal: Whether or not ot print iterations

        Returns:
            A: Factor loading matrix
            psi: uniqueness
        '''
        V = self.eigvals
        A = V[:, :n_comps] 
        psi = np.maximum(np.diag(self.S) - np.diag(np.dot(A, A.T)), 0.1)
        Sh = linalg_utils.replace_diagonal(self.S, psi)
        old_h = np.diag(Sh)
        ll_hist = []
        for i in range(n_iters):
            u, V = np.linalg.eig(Sh)
            A = np.dot(V[:, :n_comps], np.diag(np.sqrt(np.maximum(u[:n_comps],
                       0))))
            Sh = linalg_utils.replace_diagonal(Sh, np.diag((A**2).sum(axis=1)))
            if vocal:
                print(i)
            if np.linalg.norm(old_h - np.diag(Sh)) < tol:
                break
            old_h = np.diag(Sh)
            ll_hist.append(self.loglike(self.implied_cov(A, psi=psi)))
        psi = 1-np.diag(Sh)
        return A, psi, ll_hist


    def _lawley_ml(self, n_comps, n_iters=100, tol=1e-6, vocal=False):
        '''
        Lawley's traditional maximum likelihood estimation of factor analysis.

        Parameters:
            R: Correlation matrix
            A: Initial loadings matrix, normally principal components
            psi: Initial estimate of uniqueness
            ncomps: Number of factors
            n_iters: Number of iterations
            tol: Tolerance before convergance
            vocal: Whether or not ot print iterations

        Returns:
            A: Factor loading matrix
            psi: Uniqueness
            ll: log likelihood without the -n/2 term
        '''
        S, V = self.S, self.eigvals
        A = V[:, :n_comps] 
        psi = np.maximum(np.diag(self.S) - np.diag(np.dot(A, A.T)), 0.1)
        old_psi = -np.inf
        ll_hist = []
        for i in range(n_iters):
            Psih = np.diag(np.sqrt(1/psi))
            Sh = linalg_utils.mdot([Psih, S, Psih])
            u, V = np.linalg.eig(Sh)
            A = np.dot(np.sqrt(np.diag(psi)), np.dot(V[:, :n_comps],
                       np.diag(np.sqrt(np.maximum(u[:n_comps] - 1, 1e-9)))))
            psi = np.diag(S - np.dot(A, A.T))
            if np.linalg.norm(psi - old_psi) < tol:
                break
            if vocal:
                print(i, np.linalg.norm(psi - old_psi))
            old_psi = psi
            ll_hist.append(self.loglike(self.implied_cov(A, psi=psi)))
            
        return A, psi, ll_hist

    def _expectation_max(self, n_comps, n_iters=1000, tol=1e-9, 
                            vocal=False):
        '''
        Rubin and Thayer's expectation maximization algorithm for factor analysis

        Parameters:
            R: Correlation matrix
            A: Initial loadings matrix, normally principal components
            psi: Initial estimate of uniqueness
            ncomps: Number of factors
            n_iters: Number of iterations
            tol: Tolerance before convergance
            vocal: Whether or not ot print iterations

        Returns:
            A: Factor loading matrix
            psi: Uniqueness
            llhist: log likelihood for each iteration without the -n/2 term
        '''
        S, V = self.S, self.eigvals
        A = V[:, :n_comps]
        Ip = np.eye(n_comps)
        psi = np.maximum(np.diag(S) - np.diag(np.dot(A, A.T)), 0.1)
        prev_ll = -np.inf
        ll_hist = []
        for i in range(n_iters):
            Inv_Psi = np.diag(1.0 / psi)
            F = np.dot(Inv_Psi, A)
            G = np.dot(S, F)
            H = np.dot(G, np.linalg.inv(Ip + np.dot(A.T, F)))
            A = np.dot(G, np.linalg.inv(Ip + np.dot(H.T, F)))
            psi = np.diag(S - np.dot(H, A.T))
            ll = self.loglike(self.implied_cov(A, psi=psi))
            ll_hist.append(ll)
            if abs(ll - prev_ll) < tol:
                break
            if vocal:
                print(i, abs(ll - prev_ll))
            prev_ll = ll
        return A, psi, ll_hist

    def _compute_factors(self, A, X=None):
        '''
        Compute factors from a given loading matrix

        Parameters:
            S: Correlation matrix of data X
            A: Loadings matrix
            X: Data

        Returns:
            F: Estimated factors
        '''
        if X is None:
            X = self.data
        beta = np.dot(self.Sinv, A)
        F = np.dot(X, beta)
        return F

    def _fa_standard_errors(self, loadings, psi, R, InvSigma):
        Dp = linalg_utils.dmat(R.shape[0])
        Np = linalg_utils.nmat(R.shape[0])
        gL = Dp.T.dot(Np.dot(np.kron(loadings, np.eye(R.shape[0]))))
        gP = linalg_utils.pre_post_elim(np.eye(np.product(R.shape)))
        g = np.block([gL, gP])
        W = linalg_utils.pre_post_elim(np.kron(InvSigma, InvSigma))
        Iexp = linalg_utils.mdot([g.T, W, g])
        Vcov = np.linalg.pinv(Iexp)
        SE_params = np.sqrt(np.diag(Vcov/self.n))
        t1 = np.product(loadings.shape)
        SE_loadings = linalg_utils.nvec(SE_params[:t1], *loadings.shape)
        SE_psi = np.diag(linalg_utils.invech(SE_params[t1:]))
        return SE_loadings, SE_psi

    def fit(self, n_comps, method='EM', n_iters=1000, tol=1e-12,
            rotation=None, vocal=False, custom_gamma=None,
            n_rotation_iters=500, constraints=None, k=4,
            standard_errors=True):
        '''
        Fitting of the factor analysis model

        Parameters:
            ncomps: The number of components (latent factors) to use
            method: method of factor extraction, default is expectation max
            rotation: whether or not to rotate
            vocal: whether or not to print algorithmic details
            custom_gamma: to use when obliquely rotating
            n_rotation_iters: the number of iterations for the GPA algorithm
            '''

        if method == 'EM':
            A, psi, ll_hist = self._expectation_max(n_comps, vocal=vocal, 
                                                    n_iters=n_iters, tol=tol)
        if method == 'ML':
            A, psi, ll_hist = self._lawley_ml(n_comps, vocal=vocal, 
                                              n_iters=n_iters)
        if method == 'PAF':
            A, psi = self._pfa(n_comps, vocal=vocal, n_iters=n_iters)

        self.Sigma = np.dot(A, A.T) + np.diag(psi)
        communalities = 1 - psi
        factors = self._compute_factors(A, self.data)
        if rotation is not None:
            L, _ = linalg_utils.rotate(A, rotation, custom_gamma=custom_gamma,
                                       n_iters=n_rotation_iters, k=k)
            rotated_factors = self._compute_factors(L, self.data)

        else:
            L = None
            rotated_factors = None
        if self.is_pd:
            labels = ['Factor %i' % i for i in range(self.ncomps)]
            self.loadings = pd.DataFrame(A, index=self.cols, columns=labels)
            self.psi = pd.DataFrame(psi, index=self.cols)
            self.communalities = pd.DataFrame(communalities, index=self.cols)
            self.factors = pd.DataFrame(factors, index=self.ix, columns=labels)
            if L is not None:
                self.rotated_loadings = pd.DataFrame(L, index=self.cols,
                                                     columns=labels)
                self.rotated_factors = pd.DataFrame(rotated_factors,
                                                    index=self.ix)
                tR = np.trace(self.S)
            self.expvar = pd.DataFrame((self.loadings**2).sum(axis=0)/tR)
            self.Rh = pd.DataFrame(self.Sigma,
                                   index=self.cols,
                                   columns=self.cols)
        else:
            self.loadings = A
            self.psi = psi
            self.communalities = communalities
            self.factors = factors
            self.rotated_loadings = L
            self.rotated_factors = rotated_factors
            self.expvar = (self.loadings**2).sum(axis=0) / np.trace(self.S)
            self.Rh = np.dot(A, A.T) + np.diag(psi)
        self.ll_hist = ll_hist
        Sigma, sigcols, sigx, spd = check_type(self.Rh)
        S, scols, sx, spd = check_type(self.R)
        InvSigma = np.linalg.inv(Sigma)
        n, p, k = self.n, self.p, self.ncomps
        self.ll_full = -n/2*(np.linalg.slogdet(Sigma)[1]+np.trace(np.dot(S,
                             np.linalg.inv(Sigma))))
        self.df = p*k+p-k**2
        self.G1 = n*(np.trace(S.dot(InvSigma))
                     - np.log(np.det(S.dot(InvSigma))) - p)
        self.G2 = n*(0.5*np.trace((Sigma - S).dot(np.linalg.inv(S)))**2)
        self.G3 = n*(0.5*np.trace((Sigma - S).dot(np.linalg.InvSigma))**2)
        self.chi2 = np.array([[self.G1, self.G2, self.G3]])
        self.AIC = 2 * self.df-2 * self.ll_full
        self.BIC = np.log(n) * self.df - 2 * self.ll_full
        self.chi2_pval = sp.stats.chi2.sf(self.chi2, self.df)
        self.SMSR = statfunc_utils.SRMR(Sigma, S, self.df)
        upper = SimpleTable([[self.AIC, self.BIC, self.df]],
                            headers=['AIC', 'BIC', 'df'],
                            stubs=['ll stats'],
                            data_fmts=['%8.3f'])
        lower = SimpleTable(np.concatenate([self.chi2, self.SMSR, self.RMSEA],
                                           axis=0),
                            headers=['Chi2', 'SMSR', 'RMSEA'],
                            stubs=['G1', 'G2', 'G3'],
                            data_fmts=['%8.3f'])

        self.fit_summary = upper
        self.fit_stats = lower
        if standard_errors is True:
            args = self.loadings, np.diag(psi), self.R, InvSigma
            self.SE_loadings, self.SE_Psi = self._fa_standard_errors(*args)

            if self.cols is not None:
                self.SE_loadings = pd.DataFrame(self.SE_loadings,
                                                index=self.cols,
                                                columns=labels)
                self.SE_Psi = pd.DataFrame(self.SE_Psi, index=self.cols)

            self.loading_tvals = self.loadings/self.SE_loadings
            self.psi_tvals = self.psi/self.SE_Psi


class CFA:

    def __init__(self, X):
        '''
        Confirmatory factor analysis.  Differes from EFA by containing
        a matrix of factor correlations, phi, as well as option for a priori
        constraints

        If blockwise constraints are applied to loadings and factor
        correlations, this is equivalent to the general SEM measurement
        model

        Parameters:
            X: Matrix of data to be factor analyzed
        '''
        self.X, self.cols, self.ix, self.is_pd = check_type(X)
        self.R = cov(X)
        self.eigvals, self.eigvecs = linalg_utils.sorted_eigh(self.R)
        self.n, self.p = X.shape

    def _cfa_ll(self, A, Psi, Phi, Cyy, ll_const):
        Sigma = linalg_utils.mdot([A, Phi, A.T]) + Psi
        InvSg = np.linalg.pinv(Sigma)
        ll = np.log(np.linalg.det(Sigma)) + np.trace(np.dot(Cyy, InvSg))
        ll += ll_const
        return ll

    def em_cfa(self, A, Psi, Cyy, Phi=None, constraints=None, n_iters=1000,
               tol=1e-9,
               estimate_factor_correlations=True):
        ll_prev = np.inf
        ll_hist = []
        if constraints is None:
            constraints = np.ones(A.shape)
        if Phi is None:
            Phi = linalg_utils.normalize_diag(np.eye(A.shape[1])+0.1)
        constraints = constraints.astype(bool)
        Phi_k = Phi.copy()
        p, k = A.shape
        A = A.astype(float)
        idx = np.arange(k)
        ll_const = -p-np.log(np.linalg.det(Cyy))
        for i in range(n_iters):
            Sigma = linalg_utils.mdot([A, Phi, A.T]) + Psi
            InvSg = np.linalg.pinv(Sigma)
            d = linalg_utils.mdot([InvSg, A, Phi])
            D = Phi - np.linalg.mdot([Phi, A.T, InvSg, A, Phi])
            Cyz = np.dot(Cyy, d)
            Czy = Cyz.T
            Czz = linalg_utils.mdot([d.T, Cyy, d]) + D
            G = np.linalg.pinv(Czz)
            A_new = np.zeros(A.shape).astype(float)
            Psi_new = np.zeros(Cyy.shape)
            for j in range(p):
                ix = constraints[j]
                A_new[j][idx[ix]] = G[ix][:, ix].dot(Czy[ix, j])
                Psi_new[j, j] = Cyy[j, j] - Cyz[j, ix].dot(A[j, ix])
            Phi_new = linalg_utils.normalize_diag(Czz)
            if estimate_factor_correlations is False:
                Phi_new = Phi_k
            ll = self._cfa_ll(A_new, Psi_new, Phi_new, Cyy, ll_const)
            ll_hist.append(ll)
            ll_diff = ll - ll_prev
            if ((ll_diff > 0) | (abs(ll_diff) < tol)):
                break
            A, Phi, Psi, ll_prev = A_new, Phi_new, Psi_new, ll
        return A, Psi, Phi, ll_hist

    def fa_standard_errors(self, L, Phi, Psi, S, InvSigma):
        p, p = S.shape
        Dp = linalg_utils.dmat(p)
        gL = Dp.T.dot(linalg_utils.nmat(p).dot(np.kron(np.dot(L, Phi),
                                               np.eye(p))))
        gF = linalg_utils.pre_post_elim(np.kron(L, L))
        gP = linalg_utils.pre_post_elim(np.eye(np.product(S.shape)))
        g = np.block([gL, gF, gP])
        W = linalg_utils.pre_post_elim(np.kron(InvSigma, InvSigma))
        ncov = linalg_utils.mdot([g.T, W, g])
        vcov = (np.linalg.pinv(ncov) / self.n)
        SE_params = np.sqrt(np.diag(vcov)/self.n)
        SE_loadings = linalg_utils.invec(SE_params[:gL.shape[1]], *L.shape)
        SE_phi = linalg_utils.invech(SE_params[gL.shape[1]:gL.shape[1]
                                               + gF.shape[1]])
        SE_psi = linalg_utils.diag2(linalg_utils.invech(SE_params[gL.shape[1]
                                    + gF.shape[1]:]))
        return SE_loadings, SE_phi, SE_psi

    def fit(self, ncomps, loadings_init=None, factor_correlations=None,
            estimate_factor_correlations=True, constraints=None,
            n_iters=1000, tol=1e-12, SEs=False):
        if loadings_init is None:
            self.A_init = np.dot(self.eigvecs[:, :ncomps],
                                 np.diag(np.sqrt(self.eigvals[:ncomps])))
        else:
            self.A_init = loadings_init
        self.Psi_init = self.R - np.dot(self.A_init, self.A_init.T)

        self.loadings, self.uniqueness, self.factor_correlations, \
        self.ll_hist = self.em_cfa(A=self.A_init, Psi=self.Psi_init,
                              Cyy=self.R, Phi=factor_correlations,
                              constraints=constraints, 
                              n_iters=n_iters, tol=tol,
                              estimate_factor_correlations=estimate_factor_correlations)
        
        self.factors = linalg_utils.mdot([self.X, np.linalg.inv(self.R), self.loadings])
       
        self.psi = self.uniqueness
        self.phi = self.factor_correlations
        self.Rh = linalg_utils.mdot([self.loadings, self.factor_correlations, 
                             self.loadings.T]) + self.psi
        if SEs is True:
            args = self.loadings, self.factor_correlations, self.psi, self.R, np.linalg.inv(self.Rh)
            self.SE_Loadings, self.SE_Psi, self.SE_Phi = self.fa_standard_errors(*args)
            if self.is_pd:
                self.SE_loadings = pd.DataFrame(self.SE_Loadings, index=self.cols)

        if self.is_pd:
            self.loadings = pd.DataFrame(self.loadings, index=self.cols)
            self.factors = pd.DataFrame(self.factors, index=self.ix)
            self.Rh = pd.DataFrame(self.Rh, index=self.cols, columns=self.cols)
        if SEs is True:
            self.loading_tvals = self.loadings/self.SE_Loadings
        
        Sigma, sigcols, sigx, spd = check_type(self.Rh)
        S, scols, sx, spd = check_type(self.R)
        InvSigma = np.linalg.inv(Sigma)
        n, p, k = self.n, self.p, ncomps
        
        if constraints is None:
            q = 0
        else:
            q = np.sum(constraints)
        
        self.ll_full = -n/2*(np.log(np.linalg.det(Sigma))+np.trace(np.dot(S, np.linalg.inv(Sigma)))\
                             - np.log(np.linalg.det(S))-p)
        self.df = 0.5*p*(p+1.0)-p*k-0.5*k*(k+1)-p+q+np.max([q, k])
        
        self.G1 = n*(np.trace(S.dot(InvSigma)) + np.log(np.linalg.det(Sigma))\
                     - np.log(np.linalg.det(S))-p)
        self.G2 = n*(0.5*np.trace((Sigma - S).dot(np.linalg.inv(S)))**2)
        self.G3 = n*(0.5*np.trace((Sigma - S).dot(InvSigma))**2)
        
        self.chi2 = np.array([[self.G1, self.G2, self.G3]])
        self.RMSEA = np.sqrt(np.maximum(self.chi2-self.df, 0)/(self.df*(n-1)))

        self.SMSR = statfunc_utils.srmr(Sigma, S, self.df)
        self.AIC =2*self.df-2*self.ll_full
        self.BIC = np.log(n) * self.df - 2*self.ll_full
        self.chi2_pval = sp.stats.chi2.sf(self.chi2, self.df)

        upper = SimpleTable([[self.AIC, self.BIC, self.df]],
                            headers=['AIC', 'BIC', 'df'],
                            stubs=['ll stats'],
                            data_fmts=['%8.3f'])
        
        lower=SimpleTable(np.concatenate([self.chi2,self.SMSR,self.RMSEA],
                                         axis=0),
                            headers=['Chi2', 'SMSR', 'RMSEA'],
                            stubs=['G1', 'G2', 'G3'],
                            data_fmts=['%8.3f'])
        
        self.fit_summary = upper
        self.fit_stats = lower
    




class FactorAnalysis:
    
    def __init__(self, X, nfacs=None, orthogonal=True, unit_var=True):
        if nfacs is None:
            nfacs = X.shape[1]
        self.X, self.xcols, self.xix, self.is_pd = check_type(X)
        self.S = cov(X)
        U, self.V = np.linalg.eigh(self.S)
        self.U = np.diag(U)
        self.n, self.p = X.shape
        self.q = nfacs
        
        self.Lambda = self.V[:, :nfacs]
        if orthogonal is True:
            self.Phi = np.eye(self.q)
        else:
            self.Phi = np.eye(self.q) + linalg_utils.jmat(self.q,self.q) / 20.0\
                       - np.eye(self.q)/20.0
        self.Psi = np.diag((self.V**2)[:, :nfacs].sum(axis=1))
        if unit_var is True:
            Phi = self.Phi.copy() - np.eye(self.q)
        else:
            Phi = self.Phi.copy()
        self.params = np.block([linalg_utils.vec(self.Lambda), linalg_utils.vech(self.Phi), 
                                linalg_utils.vech(self.Psi)])
        self.idx =  np.block([linalg_utils.vec(self.Lambda), linalg_utils.vech(Phi), 
                                linalg_utils.vech(self.Psi)]) !=0
        bounds = [(None, None) for i in range(self.p*self.q)]
        bounds+= [(0, None) if x==1 else (None, None) for x in linalg_utils.vech(np.eye(self.q))]
        bounds+= [(0, None) if x==1 else (None, None) for x in linalg_utils.vech(np.eye(self.p))]
        bounds =  np.array(bounds)[self.idx]
        bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds = bounds
        self.free = self.params[self.idx]
        
    def p2m(self, params):
        Lambda = linalg_utils.invec(params[:self.p*self.q], self.p, self.q)
        Phi = linalg_utils.invech(params[int(self.p*self.q):int(self.p*self.q+(self.q+1)*self.q/2)])
        Psi = linalg_utils.invech(params[int(self.p*self.q+(self.q+1)*self.q/2):])
        return Lambda, Phi, Psi
    
    def loglike(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = linalg_utils.mdot([Lambda, Phi, Lambda.T]) + Psi**2
        return np.linalg.slogdet(Sigma)[1]+np.trace(np.dot(np.linalg.pinv(Sigma), S))
    
    
    def gradient(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = linalg_utils.mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = np.linalg.pinv(Sigma)
        R = Sigma - S
        VRV = V.dot(R).dot(V)
        g = np.block([2*linalg_utils.vec(linalg_utils.mdot([VRV, Lambda, Phi])),
                      linalg_utils.vech(linalg_utils.mdot([Lambda.T, VRV, Lambda])),
                      2*linalg_utils.vech(VRV.dot(Psi))])
        return g[self.idx]
        
    
    def hessian(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = linalg_utils.mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = np.linalg.pinv(Sigma)
        Np = linalg_utils.nmat(self.p)
        Ip = linalg_utils.eye(self.p)
        #Ip2 = eye(self.p*self.p)
        Dp = linalg_utils.dmat(self.p)
        J = np.block([2 * Dp.T.dot(Np.dot(np.kron(np.dot(Lambda, Phi), Ip))),
                      linalg_utils.pre_post_elim(np.kron(Lambda, Lambda)),
                      2*linalg_utils.pre_post_elim(Np.dot(np.kron(Psi, Ip)))])
        Q0 = np.kron(linalg_utils.mdot([V, (S - Sigma), V]), V)
        Q1 = np.kron(V, linalg_utils.mdot([V, S, V]))
        Q = linalg_utils.pre_post_elim(Q0 + Q1)
        H = linalg_utils.mdot([J.T, Q, J])
        H = H[self.idx][:, self.idx]
        return H
    
    def dsigma(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = linalg_utils.mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = np.pinv(Sigma)
        Np = linalg_utils.nmat(self.p)
        Ip = np.eye(self.p)
        #Ip2 = eye(self.p*self.p)
        Dp = linalg_utils.dmat(self.p)
        J = np.block([2 * Dp.T.dot(Np.dot(np.kron(np.dot(Lambda, Phi), Ip))),
                      linalg_utils.pre_post_elim(np.kron(Lambda, Lambda)),
                      2*linalg_utils.pre_post_elim(Np.dot(np.kron(Psi, Ip)))])
        Q0 = np.kron(linalg_utils.mdot([V, (S - Sigma), V]), V)
        Q1 = np.kron(V, linalg_utils.mdot([V, S, V]))

        return J, linalg_utils.pre_post_elim(Q0 + Q1)
    
    def fit(self, optimizer_kwargs=None):
        if optimizer_kwargs is None:
            optimizer_kwargs = {'method':'trust-constr',
                                'options':{'verbose':0}}
        #Hessian based optimization is less efficient
        optimizer = minimize(self.loglike, self.free, jac=self.gradient,
                     bounds=self.bounds, 
                     **optimizer_kwargs)
        self.free = optimizer.x
        self.params[self.idx] = self.free
        self.Lambda, self.Phi, self.Psi = self.p2m(self.params)
        
        self.Sigma = linalg_utils.mdot([self.Lambda, self.Phi, self.Lambda.T]) + self.Psi**2
        self.SE =  np.diag(np.linalg.pinv(self.n*self.hessian(self.free)))**0.5
        self.optimizer = optimizer
        self.res = np.block([self.free[:, None], 
                             self.SE[:, None], (self.free/self.SE)[:, None]])
        self.res = pd.DataFrame(self.res, columns=['coefficient', 'SE', 't'])
        self.res['p value'] = sp.stats.t.sf(np.abs(self.res['t']), self.X.shape[0])
        if self.is_pd:
            cols = ['Factor %i'%i for i in range(1, int(self.Lambda.shape[1]+1.0))]
            self.Lambda = pd.DataFrame(self.Lambda, index=self.xcols,
                                           columns=cols)
            self.Phi = pd.DataFrame(self.Phi, columns=cols, index=cols)
            self.Psi = pd.DataFrame(self.Psi, index=self.xcols,
                                    columns=self.xcols)
        self.chi2 = np.linalg.slogdet(self.Sigma)[1] \
                    + np.trace(np.dot(np.linalg.pinv(self.Sigma), 
                            self.S)) - self.p - np.linalg.slogdet(self.S)[1]
       
        t = (self.p + 1.0) * self.p
        self.df = t  - np.sum(self.idx)
        self.GFI = statfunc_utils.gfi(self.Sigma, self.S)
        self.AGFI = statfunc_utils.agfi(self.Sigma, self.S, self.df)
        self.stdchi2 = (self.chi2 - self.df) /  np.sqrt(2*self.df)
        self.RMSEA = np.sqrt(np.maximum(self.chi2-self.df, 0)/(self.df*(self.n-1)))
        self.SRMR = statfunc_utils.srmr(self.Sigma, self.S, self.df)
        self.chi2, self.chi2p = statfunc_utils.lr_test(self.Sigma, self.S, self.df)
        self.sumstats = pd.DataFrame([[self.AGFI, '-'],
                                      [self.GFI, '-'],
                                      [self.RMSEA, '-'],
                                      [self.SRMR, '-'],
                                      [self.chi2, self.chi2p],
                                      [self.stdchi2, '-']
                                      ])
        self.sumstats.index = ['AGFI', 'GFI', 'RMSEA', 'SRMR',
                               'chi2', 'chi2_standard']
        self.sumstats.columns=['Goodness_of_fit', 'P value']

    
