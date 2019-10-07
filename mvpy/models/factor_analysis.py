#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:46:32 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
from numpy import diag, sqrt, dot, kron, log, trace, eye, ones

from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize
from numpy.linalg import pinv, eig, norm, det, inv, eigh, slogdet
from statsmodels.iolib.table import SimpleTable

from ..utils import statfunc_utils
from ..utils.base_utils import corr, cov, check_type
from ..utils.linalg_utils import (sorted_eig, mdot, near_psd, diag2, 
                            replace_diagonal, pre_post_elim, dmat, nmat,
                            invec, invech, multi_corr, rotate, sorted_eigh, 
                            normalize_diag, jmat, vec, vech)

class EFA:
    
    def __init__(self, data, init_psi='multi_corr', method='corr',
                 n_iters=1000):
        if method=='corr':
            R = corr(data)
            if np.min(eig(R)[0]) < 0:
                R = near_psd(R)
        elif method=='cov':
            R = cov(data)
        u, V = sorted_eig(R)
        
        
        self.init_psi = init_psi
        self.n_iters = n_iters
        self.eig_expvar = u / u.sum()
        self.data = data
        self.R = R
        self.eigvals, self.eigvecs = u, V
        self.n, self.p = data.shape
        if type(data) is pd.DataFrame:
            self.cols = data.columns
            self.ix = data.index
        else:
            self.cols, self.ix = None, None
        
        Ri = pinv(R)
        D = diag(1.0/sqrt(diag(Ri)))
        Q = mdot([D, Ri, D])
        Ri = Ri - diag2(Ri)
        Q = Q - diag2(Q)
        
        self.MSA = norm(Ri) / (norm(Q) + norm(Ri))
        
    
    
    def _fa_log_likelihood(self, R, A, psi):
        '''
        Computes the log likelihood of a factor analysis model
        
        Parameters:
            R: Correlation matrix
            A: Loadings matrix
            psi: uniqueness, the complement of communalities
        
        Returns:
            ll: log likelihood without the -n/2 term
        '''
        Sigma = dot(A, A.T) + diag(psi)
        lnS = log(det(Sigma))
        ll = trace(dot(Sigma, R)) + lnS
        return ll
    
    def _fa_principal_factor(self, R, A, psi, ncomps, n_iters=100, tol=1e-6, vocal=False):
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
        Rh = replace_diagonal(R, psi)
        old_h = diag(Rh)
        for i in range(n_iters):
            u, V = eig(Rh)
            A = dot(V[:, :ncomps], diag(sqrt(np.maximum(u[:ncomps], 0))))
            Rh = replace_diagonal(Rh, diag((A**2).sum(axis=1)))
            if vocal==True:
                print(i)
            if norm(old_h - diag(Rh)) < tol:
                break
            old_h = diag(Rh)
        psi = 1-diag(Rh)
        return A, psi
    
    
    def _fa_maximum_likelihood(self, R, A, psi, ncomps, n_iters=100, tol=1e-6, vocal=False):
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
        old_psi = -np.inf
        for i in range(n_iters):
            Psih = diag(sqrt(1/psi))
            Rh = mdot([Psih, R, Psih])
            u, V = eig(Rh)
            A = dot(sqrt(diag(psi)), dot(V[:, :ncomps], diag(sqrt(np.maximum(u[:ncomps] - 1, 1e-9)))))
            psi = diag(R - dot(A, A.T))
            if norm(psi - old_psi) < tol:
                break
            if vocal==True:
                print(i, norm(psi - old_psi))
            old_psi = psi
        ll = self._fa_log_likelihood(R, A, psi)
        return A, psi, ll
    
    def _fa_expectation_max(self, R, A, psi, ncomps, n_iters=1000, tol=1e-9, vocal=False,
                        constraints=None):
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
        A = A[:, :ncomps]
        I = eye(ncomps)
        prev_ll = -np.inf
        ll_hist = []
        for i in range(n_iters):
            Inv_Psi = diag(1 / psi)
            F = dot(Inv_Psi, A)
            G = dot(R, F)
            H = dot(G, inv(I + dot(A.T, F)))
            A = dot(G, inv(I + dot(H.T, F)))
            if constraints is not None:
                A = A * constraints
            psi = diag(R - dot(H, A.T))
            ll = self._fa_log_likelihood(R, A, psi)
            ll_hist.append(ll)
            if abs(ll - prev_ll) < tol:
                break
            if vocal==True:
                print(i, abs(ll - prev_ll))
            prev_ll = ll
        return A, psi, pd.DataFrame(ll_hist)
    
    def _compute_factors(self, R, A, X):
        '''
        Compute factors from a given loading matrix
        
        Parameters:
            R: Correlation matrix of data X
            A: Loadings matrix
            X: Data
        
        Returns:
            F: Estimated factors
        '''
        beta = dot(pinv(R), A)
        F = dot(X, beta)
        return F
    
    def _fa_standard_errors(self, loadings, psi, R, InvSigma):
        Dp = dmat(R.shape[0])
        Np = nmat(R.shape[0])
        gL = Dp.T.dot(Np.dot(kron(loadings, eye(R.shape[0]))))
        gP = pre_post_elim(eye(np.product(R.shape)))
        g = np.block([gL, gP])
        Iexp = mdot([g.T, pre_post_elim(kron(InvSigma, InvSigma)), g])
        Vcov = pinv(Iexp)
        SE_params = sqrt(diag(Vcov/self.n))
        t1 = np.product(loadings.shape)
        SE_loadings = invec(SE_params[:t1], *loadings.shape)
        SE_psi = diag(invech(SE_params[t1:]))
        return SE_loadings, SE_psi
        
    def fit(self, ncomps, method='EM', n_iters=1000, tol=1e-12, 
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
        R, data, init_psi, u, V = self.R, self.data, self.init_psi, \
                                    self.eigvals, self.eigvecs
                                    
        A = dot(V[:, :ncomps], diag(sqrt(u[:ncomps])))

        if init_psi == 'multi_corr':
            psi = multi_corr(R)
            
        if init_psi == 'max_corr':
            psi = np.max(abs(R - eye(len(R))), axis=0)
            
        if init_psi == 'pca':
            psi = np.maximum(diag(R) - diag(dot(A, A.T)), 0.1)
            
        self.ncomps = ncomps
        self.A, self.psi = A, psi
        self.constraints = constraints
        
        if method=='EM':
            A, psi, ll_hist = self._fa_expectation_max(R, A, psi, ncomps,
                                              vocal=vocal, 
                                              n_iters=self.n_iters,
                                              tol=tol,
                                              constraints=self.constraints)
        if method=='ML':
            A, psi, ll_hist = self._fa_maximum_likelihood(R, A, psi, ncomps,
                                                 vocal=vocal,
                                                 n_iters=self.n_iters)
        if method=='PAF':
            A, psi = self._fa_principal_factor(R, A, psi, ncomps, vocal=vocal,
                                      n_iters=self.n_iters)
            ll_hist =self. _fa_log_likelihood(R, A, psi)
         
        self._Sigma = dot(A, A.T) + diag(psi)
        communalities = 1 - psi
        factors = self._compute_factors(R, A, data)
        if rotation is not None:
            L, _ = rotate(A, rotation, custom_gamma=custom_gamma, 
                          n_iters=n_rotation_iters, k=k)
            rotated_factors = self._compute_factors(R, L, data)
            
        else:
            
            L = None
            rotated_factors = None
    
        if (self.cols is not None):
            
            labels =['Factor %i'%i for i in range(self.ncomps)]
            self.loadings = pd.DataFrame(A, index=self.cols, columns=labels)
            self.psi = pd.DataFrame(psi, index=self.cols)
            self.communalities = pd.DataFrame(communalities, index=self.cols)
            self.factors = pd.DataFrame(factors, index=self.ix, columns=labels)
            if L is not None:
                
                self.rotated_loadings = pd.DataFrame(L, index=self.cols,
                                                     columns=labels)
                self.rotated_factors = pd.DataFrame(rotated_factors, 
                                                    index=self.ix)
                
            self.expvar = pd.DataFrame((self.loadings**2).sum(axis=0)/trace(R))
            self.Rh = pd.DataFrame(dot(A, A.T) + diag(psi), index=self.cols,
                                   columns=self.cols)
            
        else:
            
            self.loadings = A
            self.psi = psi
            self.communalities = communalities
            self.factors = factors
            self.rotated_loadings = L
            self.rotated_factors = rotated_factors
            self.expvar = (self.loadings**2).sum(axis=0) / trace(R)
            self.Rh = dot(A, A.T) + diag(psi)
        self.ll_hist = ll_hist
        
        Sigma, sigcols, sigx, spd = check_type(self.Rh)
        S, scols, sx, spd = check_type(self.R)
        InvSigma = inv(Sigma)
        n, p, k = self.n, self.p, self.ncomps
       
        
        self.ll_full = -n/2*(log(det(Sigma))+trace(dot(S, inv(Sigma))))
        self.df = p*k+p-k**2
        
        self.G1 = n*(trace(S.dot(InvSigma)) - log(det(S.dot(InvSigma))) - p)
        self.G2 = n*(0.5*trace((Sigma - S).dot(inv(S)))**2)
        self.G3 = n*(0.5*trace((Sigma - S).dot(InvSigma))**2)
        
        self.chi2 = np.array([[self.G1, self.G2, self.G3]])
        self.RMSEA = sqrt(np.maximum(self.chi2-self.df, 0)/(self.df*(n-1)))
        Rij = diag(S)[:, None].dot(diag(S)[:, None].T)
        self.SMSR = 2*np.sum(np.sum((S - Sigma)**2 / Rij))/(p**2+p)
        self.AIC =2*self.df-2*self.ll_full
        self.BIC = log(n) * self.df - 2*self.ll_full
        self.chi2_pval = 1 - chi2_dist.cdf(self.chi2, self.df)
        self.SMSR = np.array([[self.SMSR]*3])
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
        if standard_errors is True:
            args = self.loadings, diag(psi), self.R, InvSigma
            self.SE_loadings, self.SE_Psi = self._fa_standard_errors(*args)
            
            if self.cols is not None:
                self.SE_loadings = pd.DataFrame(self.SE_loadings, index=self.cols,
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
        self.eigvals, self.eigvecs = sorted_eigh(self.R)
        self.n, self.p = X.shape
    
    
    def _cfa_ll(self, A, Psi, Phi, Cyy, ll_const):
        Sigma = mdot([A, Phi, A.T]) + Psi
        InvSg = pinv(Sigma)
        ll = log(det(Sigma)) + trace(dot(Cyy, InvSg)) + ll_const
        return ll
    
    def em_cfa(self, A, Psi, Cyy, Phi=None, constraints=None, n_iters=1000, tol=1e-9,
               estimate_factor_correlations=True):
        ll_prev = np.inf
        ll_hist = []
        if constraints is None:
            constraints = ones(A.shape)
        if Phi is None:
            Phi = normalize_diag(eye(A.shape[1])+0.1)
        constraints = constraints.astype(bool)
        Phi_k = Phi.copy()
        p, k = A.shape
        A = A.astype(float)
        idx = np.arange(k)
        ll_const = -p-log(det(Cyy))
        for i in range(n_iters):
            Sigma = mdot([A, Phi, A.T]) + Psi
            InvSg = pinv(Sigma)
            d = mdot([InvSg, A, Phi])
            D = Phi - mdot([Phi, A.T, InvSg, A, Phi])
            Cyz = dot(Cyy, d)
            Czy = Cyz.T
            Czz = mdot([d.T, Cyy, d]) + D
            G = pinv(Czz)
            A_new = np.zeros(A.shape).astype(float)
            Psi_new = np.zeros(Cyy.shape)
            for j in range(p):
                ix = constraints[j]
                A_new[j][idx[ix]] = G[ix][:, ix].dot(Czy[ix, j])
                Psi_new[j, j] = Cyy[j, j] - Cyz[j, ix].dot(A[j, ix])
            Phi_new = normalize_diag(Czz)
            if estimate_factor_correlations is False:
                Phi_new = Phi_k
            ll = self._cfa_ll(A_new, Psi_new, Phi_new, Cyy, ll_const)
            ll_hist.append(ll)
            ll_diff = ll - ll_prev
            if ((ll_diff>0)|(abs(ll_diff)<tol)):
                break
            A, Phi, Psi, ll_prev = A_new, Phi_new, Psi_new, ll
        return A, Psi, Phi, ll_hist
    
    def fa_standard_errors(self, L, Phi, Psi, S, InvSigma):
        p, p = S.shape
        Dp = dmat(p)
        gL = Dp.T.dot(nmat(p).dot(kron(dot(L, Phi), eye(p))))
        gF = pre_post_elim(kron(L, L))
        gP = pre_post_elim(eye(np.product(S.shape)))
        g = np.block([gL, gF, gP])
        ncov = mdot([g.T, pre_post_elim(kron(InvSigma, InvSigma)), g])
        vcov = (pinv(ncov) / self.n)
        SE_params = sqrt(diag(vcov)/self.n)
        SE_loadings = invec(SE_params[:gL.shape[1]], *L.shape)
        SE_phi = invech(SE_params[gL.shape[1]:gL.shape[1]+gF.shape[1]])
        SE_psi = diag2(invech(SE_params[gL.shape[1]+gF.shape[1]:]))
        return SE_loadings, SE_phi, SE_psi
    
    def fit(self, ncomps, loadings_init=None, factor_correlations=None, 
            estimate_factor_correlations=True, constraints=None,
            n_iters=1000, tol=1e-12, SEs=False):
        if loadings_init is None:
            self.A_init = dot(self.eigvecs[:, :ncomps], 
                          diag(sqrt(self.eigvals[:ncomps])))
        else:
            self.A_init = loadings_init
        self.Psi_init = self.R - dot(self.A_init, self.A_init.T)
        
        self.loadings, self.uniqueness, self.factor_correlations, \
        self.ll_hist = self.em_cfa(A=self.A_init, Psi=self.Psi_init,
                              Cyy=self.R, Phi=factor_correlations,
                              constraints=constraints, 
                              n_iters=n_iters, tol=tol,
                              estimate_factor_correlations=estimate_factor_correlations)
        
        self.factors = mdot([self.X, inv(self.R), self.loadings])
       
        self.psi = self.uniqueness
        self.phi = self.factor_correlations
        self.Rh = mdot([self.loadings, self.factor_correlations, 
                             self.loadings.T]) + self.psi
        if SEs is True:
            args = self.loadings, self.factor_correlations, self.psi, self.R, inv(self.Rh)
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
        InvSigma = inv(Sigma)
        n, p, k = self.n, self.p, ncomps
        
        if constraints is None:
            q = 0
        else:
            q = np.sum(constraints)
        
        self.ll_full = -n/2*(log(det(Sigma))+trace(dot(S, inv(Sigma)))- log(det(S))-p)
        self.df = 0.5*p*(p+1.0)-p*k-0.5*k*(k+1)-p+q+np.max([q, k])
        
        self.G1 = n*(trace(S.dot(InvSigma)) + log(det(Sigma)) - log(det(S))-p)
        self.G2 = n*(0.5*trace((Sigma - S).dot(inv(S)))**2)
        self.G3 = n*(0.5*trace((Sigma - S).dot(InvSigma))**2)
        
        self.chi2 = np.array([[self.G1, self.G2, self.G3]])
        self.RMSEA = sqrt(np.maximum(self.chi2-self.df, 0)/(self.df*(n-1)))
        Rij = diag(S)[:, None].dot(diag(S)[:, None].T)
        self.SMSR = 2*np.sum(np.sum((S - Sigma)**2 / Rij))/(p**2+p)
        self.AIC =2*self.df-2*self.ll_full
        self.BIC = log(n) * self.df - 2*self.ll_full
        self.chi2_pval = 1 - chi2_dist.cdf(self.chi2, self.df)
        self.SMSR = np.array([[self.SMSR]*3])
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
        U, self.V = eigh(self.S)
        self.U = diag(U)
        self.n, self.p = X.shape
        self.q = nfacs
        
        self.Lambda = self.V[:, :nfacs]
        if orthogonal is True:
            self.Phi = eye(self.q)
        else:
            self.Phi = eye(self.q) + jmat(self.q, self.q) / 20.0 - eye(self.q)/20.0
        self.Psi = np.diag((self.V**2)[:, :nfacs].sum(axis=1))
        if unit_var is True:
            Phi = self.Phi.copy() * 0.0
        else:
            Phi = self.Phi.copy()
        self.params = np.block([vec(self.Lambda), vech(self.Phi), 
                                vech(self.Psi)])
        self.idx =  np.block([vec(self.Lambda), vech(Phi), 
                                vech(self.Psi)]) !=0
        bounds = [(None, None) for i in range(self.p*self.q)]
        bounds+= [(0, None) if x==1 else (None, None) for x in vech(eye(self.q))]
        bounds+= [(0, None) if x==1 else (None, None) for x in vech(eye(self.p))]
        bounds =  np.array(bounds)[self.idx]
        bounds = [tuple(x) for x in bounds.tolist()]
        self.bounds = bounds
        self.free = self.params[self.idx]
        
    def p2m(self, params):
        Lambda = invec(params[:self.p*self.q], self.p, self.q)
        Phi = invech(params[int(self.p*self.q):int(self.p*self.q+(self.q+1)*self.q/2)])
        Psi = invech(params[int(self.p*self.q+(self.q+1)*self.q/2):])
        return Lambda, Phi, Psi
    
    def loglike(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = mdot([Lambda, Phi, Lambda.T]) + Psi**2
        return slogdet(Sigma)[1]+trace(dot(pinv(Sigma), S))
    
    
    def gradient(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = pinv(Sigma)
        R = Sigma - S
        VRV = V.dot(R).dot(V)
        g = np.block([2*vec(mdot([VRV, Lambda, Phi])),
                      vech(mdot([Lambda.T, VRV, Lambda])),
                      2*vech(VRV.dot(Psi))])
        return g[self.idx]
        
    
    def hessian(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = pinv(Sigma)
        Np = nmat(self.p)
        Ip = eye(self.p)
        #Ip2 = eye(self.p*self.p)
        Dp = dmat(self.p)
        J = np.block([2 * Dp.T.dot(Np.dot(kron(dot(Lambda, Phi), Ip))),
                      pre_post_elim(kron(Lambda, Lambda)),
                      2*pre_post_elim(Np.dot(kron(Psi, Ip)))])
        Q0 = kron(mdot([V, (S - Sigma), V]), V)
        Q1 = kron(V, mdot([V, S, V]))
        Q = pre_post_elim(Q0 + Q1)
        H = mdot([J.T, Q, J])
        H = H[self.idx][:, self.idx]
        return H
    
    def dsigma(self, free):
        params = self.params.copy()
        params[self.idx] = free
        S = self.S
        Lambda, Phi, Psi = self.p2m(params)
        Sigma = mdot([Lambda, Phi, Lambda.T]) + Psi**2
        V = pinv(Sigma)
        Np = nmat(self.p)
        Ip = eye(self.p)
        #Ip2 = eye(self.p*self.p)
        Dp = dmat(self.p)
        J = np.block([2 * Dp.T.dot(Np.dot(kron(dot(Lambda, Phi), Ip))),
                      pre_post_elim(kron(Lambda, Lambda)),
                      2*pre_post_elim(Np.dot(kron(Psi, Ip)))])
        Q0 = kron(mdot([V, (S - Sigma), V]), V)
        Q1 = kron(V, mdot([V, S, V]))

        return J, pre_post_elim(Q0 + Q1)
    
    def fit(self, verbose=2, n_iters=2000, gtol=1e-8, xtol=1e-9):
        #Hessian based optimization is less efficient
        optimizer = minimize(self.loglike, self.free, jac=self.gradient,
                     bounds=self.bounds, 
                     method='trust-constr', options={'verbose':verbose, 
                                                     'maxiter':n_iters,
                                                     'gtol':gtol, 
                                                     'xtol':xtol})
        self.free = optimizer.x
        self.params[self.idx] = self.free
        self.Lambda, self.Phi, self.Psi = self.p2m(self.params)
        
        self.Sigma = mdot([self.Lambda, self.Phi, self.Lambda.T]) + self.Psi**2
        self.SE =  diag(pinv(self.n*self.hessian(self.free)))**0.5
        self.optimizer = optimizer
        self.res = np.block([self.free[:, None], 
                             self.SE[:, None], (self.free/self.SE)[:, None]])
        self.chi2 = slogdet(self.Sigma)[1] + trace(dot(pinv(self.Sigma), 
                            self.S)) - self.p - slogdet(self.S)[1]
        tmp1 = pinv(self.Sigma).dot(self.S)
        tmp2 = tmp1 - eye(self.p)
        t = (self.p + 1.0) * self.p
        self.df = t  - np.sum(self.idx)
        self.GFI = statfunc_utils.gfi(self.Sigma, self.S)
        self.AGFI = statfunc_utils.agfi(self.Sigma, self.S, self.df)
        self.stdchi2 = (self.chi2 - self.df) /  sqrt(2*self.df)
        self.RMSEA = sqrt(np.maximum(self.chi2-self.df, 0)/(self.df*(self.n-1)))
        self.SRMR = statfunc_utils.srmr(self.Sigma, self.S, self.df)
	self.chi2, self.chi2p = statfunc_utils.lr_test(self.Sigma, self.S, self.df)


