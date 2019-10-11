
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:55:37 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import dot, sqrt, eye, diag, log, zeros
from numpy.linalg import multi_dot, norm, pinv
from scipy.stats import f as f_dist, chi2 as chi2_dist, t as t_dist
from scipy.stats import scoreatpercentile
from scipy.linalg import block_diag, svd

from ..utils.base_utils import (csd, check_type, corr, cov, center)
from ..utils.linalg_utils import  (inv_sqrth, sorted_eigh,
                          lstq, omat, xprod, mdot, lstq_pred, svd2)

class PLSC:
    
    def __init__(self, X, Y, center_standardize=True):
        '''
        Partial Least Squares Corrlation.  
        
        Parameters:
            X: independent or exogenous variables
            Y: dependent of endogenous variables
            center_standardize: Default True, whether or not to center and
            standardize variables
            
        '''
        
        if center_standardize is True:
            X, Y = csd(X), csd(Y)
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        self.R = corr(X, Y)
        self.n_obs = self.X.shape[0]
        
    def fit(self):
        
        X_loadings, singular_values, Y_loadings = svd2(self.R)
        X_var, Y_var = dot(self.X, X_loadings), dot(self.Y, Y_loadings)
        
        
        
        self.singular_values = singular_values
        self.inertia = self.singular_values.sum()
        
        if self.x_is_pd is True:
            self.X_loadings = pd.DataFrame(X_loadings, index=self.xcols)
            self.X_var = pd.DataFrame(X_var, index=self.xix)
        else:
            self.X_loadings = X_loadings
            self.X_var = X_var
            
        if self.x_is_pd is True:
            self.Y_loadings = pd.DataFrame(Y_loadings, index=self.ycols)
            self.Y_var = pd.DataFrame(Y_var, index=self.yix)
        else:
            self.Y_loadings = Y_loadings
            self.Y_var = Y_var
        self.permutation_test()
        self.sumstats = pd.DataFrame([[self.inertia, self.permutation_pval]],
                                     index=['Permutation Test'], 
                                     columns=['Inertia', 'P-val'])
            
    def permutation_test(self, n_permutations=1000):
        svs = []
        X, Y = self.X.copy(), self.Y.copy()
        for i in range(n_permutations):
            idx = np.random.permutation(self.n_obs)
            R_perm = corr(X[idx], Y)
            U, inertia, V = np.linalg.svd(R_perm, full_matrices=False)
            svs.append(inertia[:, None])
        self.permuted_svds = np.concatenate(svs, axis=1).T
        self.permuted_inertia = self.permuted_svds.sum(axis=1)
        self.permutation_pval = (self.inertia<self.permuted_inertia).mean()+1e-16
        
    
    def bootstrap(self, n_boot=2000):
        n = len(self.X)
        size = n
        Lx_samples, Ly_samples = [], []
        for i in range(n_boot):
            Yboot = self.Y[np.random.choice(n, size=size, replace=True)]
            Xboot = self.X[np.random.choice(n, size=size, replace=True)]
            Rboot = corr(Xboot, Yboot)
            Ub, Sb, Vb = svd2(Rboot)
            Lx_samples.append(Ub[:, :, None])
            Ly_samples.append(Vb[:, :, None])
            
        Lx_samples = np.concatenate(Lx_samples, axis=2)
        Ly_samples = np.concatenate(Ly_samples, axis=2) 
        self.Lx_SE = Lx_samples.std(axis=2)
        self.Ly_SE = Ly_samples.std(axis=2)
        
class CCA:
    
    def __init__(self, X, Y):
        '''
        Canonical Correlation Analysis
        Computes coeficients that project X and Y onto variables
        with maximum shared varianced
        
        Parameters:
            X: Matrix of n_obs by p features
            Y: Matrix of n_obs by q features
        '''
        
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(csd(X))
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(csd(Y))
        self.Sx, self.Sy, self.Sxy = (cov(self.X), cov(self.Y),
                                      cov(self.X, self.Y))
        
        self.xrank = np.linalg.matrix_rank(self.Sx)
        self.yrank = np.linalg.matrix_rank(self.Sy)
        self.rank = min(self.xrank, self.yrank)
        
        self.n_obs = X.shape[0]
        self.p, self.q = X.shape[1], Y.shape[1]
        self.min_dim = min(self.p, self.q)
        self.df1 = self.p * self.q
        self.df2 = self.n_obs - 0.5 * (self.p + self.q + 3)
        self.df3 = sqrt((self.p**2*self.q**2-4)/(self.p**2+self.q**2-5))
        self.df4 = self.df2 * self.df3 - 0.5 * self.p * self.q + 1
        self.In = eye(self.n_obs)
    
    def fit(self, method='eig'):
        if method=='eig':
            self._fit_eig()
        elif method=='svd':
            self._fit_svd()
            
        self.X_loadings = self.Sx.dot(self.X_coefs)
        self.Y_loadings = self.Sy.dot(self.Y_coefs)
        
        self._wilks_stats()
        self._repackage()
        
        corr_stats = self.rho.copy()
        corr_stats['Wilks Lambda'] = self.wilks_lambda
        corr_stats['Lawley Hotling'] = self.lawley_hotling
        corr_stats['Pillai Trace'] = self.pillai 
        corr_stats['chi2'] = self.chi2
        corr_stats['chi2p'] = self.chi2p
        corr_stats['F'] = self.F
        corr_stats['Fp'] = self.Fp
        self.sumstats = corr_stats
        
    def _boot_fit(self, X, Y):
        Sx, Sy = cov(X), cov(Y)
        Sxy = cov(X, Y)
        Sxisq = inv_sqrth(Sx)
        Syisq = inv_sqrth(Sy)
        K = multi_dot([Sxisq, Sxy, Syisq])
        U, S, V = svd2(K)
        X_coefs = dot(Sxisq, U)
        Y_coefs = dot(Syisq, V)
        rho = S
        X_loadings = Sx.dot(X_coefs)
        Y_loadings = Sy.dot(Y_coefs)
        return X_loadings, Y_loadings, rho
        
    def _fit_eig(self):
        
        self.Sxisq = inv_sqrth(self.Sx)
        self.Syisq = inv_sqrth(self.Sy)
        self.K = multi_dot([self.Sxisq, self.Sxy, self.Syisq])
        self.U, self.S, self.V = svd2(self.K)
        self.X_coefs = dot(self.Sxisq, self.U)
        self.Y_coefs = dot(self.Syisq, self.V)
        self.rho = self.S
        self.X_var,self.Y_var=dot(self.X,self.X_coefs),dot(self.Y,self.Y_coefs)
        
        
    def _fit_svd(self):
        
        self.Ux, self.Sx, self.Vx = svd2(self.X)
        self.Uy, self.Sy, self.Vy = svd2(self.Y)
        self.K = dot(self.Ux.T, self.Uy)
        self.U, self.S, self.V = svd2(self.K)
        self.X_coefs = dot(self.Vx, self.U)
        self.Y_coefs = dot(self.Vy, self.V)
        self.X_var = dot(self.X, self.X_coefs)
        self.Y_var = dot(self.Y,  self.Y_coefs)
        self.rho = self.S
        
    def _repackage(self):
        varlabel = ['Canvar %i'%i for i in range(self.min_dim)]
        if self.x_is_pd is True:
            self.X_coefs = pd.DataFrame(self.X_coefs, index=self.xcols,
                                        columns=varlabel)
            self.X_var = pd.DataFrame(self.X_var, index=self.xix,
                                      columns=varlabel)
            self.X_loadings = pd.DataFrame(self.X_loadings, index=self.xcols,
                                           columns=varlabel)
        if self.y_is_pd is True:
            
            self.Y_coefs = pd.DataFrame(self.Y_coefs, index=self.ycols,
                                        columns=varlabel)
            self.Y_var = pd.DataFrame(self.Y_var, index=self.yix,
                                          columns=varlabel)
            self.Y_loadings = pd.DataFrame(self.Y_loadings, index=self.ycols,
                                           columns=varlabel)
            
        self.rho = pd.DataFrame(diag(self.rho), index=varlabel, 
                                columns=['Canonical Correlation'])
        self.wilks_lambda = pd.DataFrame(self.wilks_lambda,
                                         index=varlabel,  
                                         columns=['Wilks Lambda'])
        self.F = pd.DataFrame(self.F, index=varlabel, columns=['F'])
        self.chi2 = pd.DataFrame(self.chi2, index=varlabel,
                                 columns=['Chi Squared'])
        self.chi2p = pd.DataFrame(self.chi2p, index=varlabel,
                                  columns=['Chi Squared P Value'])
        self.Fp = pd.DataFrame(self.Fp, index=varlabel, columns=['F pvalue'])
        self.stats = pd.concat([self.rho, self.wilks_lambda,
                                      self.chi2, self.F,
                                      self.chi2p, self.Fp], axis=1)
    def _wilks_stats(self):
        p = self.rho
        self.wilks_lambda=np.array([np.product(1-p[i:]) 
                                    for i in range(len(p))])
        self.lawley_hotling = np.array([np.sum(p[:i]**2/(1-p[:i]**2)) 
                                        for i in range(len(p))])
        self.pillai = np.array([np.sum(p[:i]**2)
                                for i in range(len(p))])
        k = self.wilks_lambda**(1/self.df3)
        self.F = (1-k)/(k) * self.df4 / self.df1
        self.chi2 = -(self.n_obs-0.5*(self.p+self.q+3))*log(self.wilks_lambda)
        self.chi2p = 1 - chi2_dist.cdf(self.chi2, self.p*self.q)
        self.Fp = 1 - f_dist.cdf(self.F, self.df1, self.df4)
    
    def plot_canvars(self, nx=1):
        fig, ax = plt.subplots(nrows=nx)
        for i in range(nx):
            ax[i].scatter(self.X_var.iloc[:, i], self.Y_var.iloc[:, i],alpha=0.5)
            txt = "$\\rho$=%4.3f; p=%4.4f"%(self.rho.iloc[i, 0], self.Fp.iloc[i])
            ax[i].annotate(txt, xytext=(-5, 5), xy=(1, 0), xycoords='axes fraction',
                           textcoords='offset points', ha='right', va='bottom')
            ymn, ymx = ax[i].get_ylim()
            xmn, xmx = ax[i].get_xlim()
            ax[i].set_ylim(ymn, ymx-0.2)
            ax[i].set_xlim(xmn, xmx+0.5)
        return fig, ax
    
    def bootstrap(self, n_samples=1000, vocal=True):
        Mx, Vx = np.zeros(self.X_loadings.shape), np.zeros(self.X_loadings.shape)
        My, Vy = np.zeros(self.Y_loadings.shape), np.zeros(self.Y_loadings.shape)
        Mb, Vb =  np.zeros((self.rho.shape[0], self.rho.shape[0])), \
                  np.zeros((self.rho.shape[0], self.rho.shape[0]))
        #Bsmp = []
        for i in range(n_samples):
            sample_ix = np.random.choice(self.n_obs, size=self.n_obs)
            X_samples = self.X[sample_ix]
            Y_samples = self.Y[sample_ix]
            XL, YL, B = self._boot_fit(X_samples, Y_samples)
            n = i+1
            if i==0:  
                Mx += XL
                My += YL
                Mb += B
            else:
                if i>=2:
                     c = (n-2) / (n - 1)
                     Vx =  c * Vx + (XL - Mx + (XL - Mx) / n)**2 / n
                     Vy =  c * Vy + (YL - My + (YL - My) / n)**2 / n
                     Vb =  c * Vb + (B - Mb + (B - Mb) / n)**2 / n
    
                Mx += (XL - Mx) / n
                My += (YL - My) / n
                Mb += (B - Mb) / n
            #Bsmp.append(B)
            self.boot_mean_LX, self.boot_mean_LY, self.boot_mean_B = Mx, My, Mb
            self.boot_var_LX, self.boot_var_LY, self.boot_var_B = Vx, Vy, Vb
            #self.Bsmp = Bsmp
            if vocal is True:
                print(i)
    
                
    
        
 
def simpls(X, Y, ncomps):
    S = dot(X.T, Y)
    n, m, q = X.shape[0], X.shape[1], Y.shape[1]
    T, P = np.zeros((n, ncomps)), np.zeros((m, ncomps))
    U, Q = np.zeros((n, ncomps)), np.zeros((q, ncomps))
    R = np.zeros((m, ncomps))
    for i in range(ncomps):
        if i==0:
            r, s, Vt = svd(S, full_matrices=False)
        else:
            Pi = P[:, :i]
            Sk = S - multi_dot([Pi, pinv(Pi.T.dot(Pi)), Pi.T, S])
            r, s, Vt = svd(Sk, full_matrices=False)
        r = r[:, [0]]
        t = dot(X, r)
        t -= t.mean()
        p = dot(X.T, t) / dot(t.T, t)
        q = dot(Y.T, t) / dot(t.T, t)
        u = dot(Y, q) 
        u -= u.mean()         
        R[:, i] = r[:, 0]
        T[:, i] = t[:, 0]
        P[:, i] = p[:, 0]
        Q[:, i] = q[:, 0]
        U[:, i] = u[:, 0]
        
    B = dot(R, Q.T)
    U = U/sqrt(norm(U, axis=0))
    Q = Q/norm(Q, axis=0)
    return T, P, U, Q, R, B



def nipals(X, Y, ncomps, n_iters=100, tol=1e-9):
    n, m, q = X.shape[0], X.shape[1], Y.shape[1]
    T, P = np.zeros((n, ncomps)), np.zeros((m, ncomps))
    U, Q = np.zeros((n, ncomps)), np.zeros((q, ncomps))
    t_old = -np.inf
    for i in range(ncomps):
        t_old = -np.inf
        for j in range(100):
            u = Y[:, [0]]
            p = dot(X.T, u) / norm(dot(X.T, u))
            t = dot(X, p)
            q = dot(Y.T, t) / norm(dot(Y.T, t))
            u = dot(Y, q)
            #b = dot(u.T, t) / dot(t.T, t)
            if norm(t - t_old)<tol:
                break
            t_old = t
        
        X = X - dot(t, p.T)
        Y = Y - dot(u, q.T)
        U[:, i] = u[:, 0]
        T[:, i] = t[:, 0]
        Q[:, i] = q[:, 0]
        P[:, i] = p[:, 0]
        B = multi_dot([P, pinv(dot(T.T, T)), T.T, U ,Q.T])
        
        
    return T, P, U, Q, B


def pls_w2a(X, Y, r, vocal=False):
    X_i, Y_i = X.copy(), Y.copy()
    Gamma, Delta = zeros((X.shape[1], r)), zeros((Y.shape[1], r))
    Xi, Omega = zeros((X.shape[0], r)), zeros((Y.shape[0], r))
    for i in range(r):
        XtY = xprod(X_i, Y_i)
        u, s, v = svd2(XtY)
        u, v = u[:, [0]], v[:, [0]]
        Xi_i, Omega_i = dot(X, u), dot(Y, v)
        gamma, delta = lstq(Xi_i, X), lstq(Omega_i, Y_i)
        X_i -= dot(Xi_i, gamma)
        Y_i -= dot(Omega_i, delta)
        Delta[:, i], Gamma[:, i] = delta, gamma
        Xi[:, [i]], Omega[:, [i]] = Xi_i, Omega_i
        if vocal==True: 
            print(i)
    Beta = lstq(Xi, Omega)
    coefs = mdot([Gamma, Beta, Delta.T])
    return Xi, Omega, Gamma, Delta, coefs

        
        

class PLSR:
    
    def __init__(self, X, Y, handle_missing='EM'):
        '''
        Partial Least Squares Regression
        Computes subspaces onto which X and Y projects which maximizes
        their shared varianced
        
        Parameters:
            X: n by p matrix of predictors
            Y: n by q matrix of endogenous variables
        '''
        
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
        
        self.n_obs = X.shape[0]
        self.nx_feats = X.shape[1]
        self.ny_feats = Y.shape[1]
        
    def fit(self, ncomps=None, method='NIPALS', n_iters=500, tol=1e-12, 
            vocal=False):
        if ncomps is None:
            ncomps = np.min([self.nx_feats, self.ny_feats])
        self.method = method
        self.ncomps=ncomps
        self.n_iters = n_iters
        self.tol = tol
        
        if method == 'NIPALS':
            self.x_factors, self.x_loadings, self.y_factors, \
            self.y_loadings, self.coefs = nipals(self.X, self.Y, self.ncomps,
                                                 n_iters=n_iters, tol=tol)
        elif method == 'SIMPLS':
            self.x_factors, self.x_loadings, self.y_factors, \
            self.y_loadings,  self.obs_coefs, self.coefs = simpls(self.X, self.Y, self.ncomps)
            
        elif method == 'W2A':
            self.x_factors, self.y_factors, self.x_loadings, \
            self.y_loadings, self.coefs = pls_w2a(self.X, self.Y, self.ncomps, 
                                                  vocal=vocal)
            
        self.Yhat = dot(self.X, self.coefs)   
        if self.x_is_pd:
            self.x_factors = pd.DataFrame(self.x_factors, index=self.xix)
            self.x_loadings = pd.DataFrame(self.x_loadings, 
                                           index=self.xcols)
            self.coefs = pd.DataFrame(self.coefs, index=self.xcols, 
                                      columns=self.ycols)
        if self.y_is_pd:
            self.y_factors = pd.DataFrame(self.y_factors, index=self.yix)
            self.y_loadings = pd.DataFrame(self.y_loadings, 
                                           index=self.ycols)
            self.Yhat = pd.DataFrame(self.Yhat, index=self.yix,
                                     columns=self.ycols)
        self.Yhat = dot(self.X, self.coefs)
    
    def bootstrap(self, n_samples=1000, method=None, 
                  n_nipals_iters=None, tol=None, CI=95, vocal=False):
        if method is None:
            method=self.method
        if n_nipals_iters is None:
            n_nipals_iters = self.n_iters
        if tol is None:
            tol = self.tol
        Mx, Vx = np.zeros(self.x_loadings.shape), np.zeros(self.x_loadings.shape)
        My, Vy = np.zeros(self.y_loadings.shape), np.zeros(self.y_loadings.shape)
        Mb, Vb =  np.zeros(self.coefs.shape), np.zeros(self.coefs.shape)
        #Bsmp = []
        for i in range(n_samples):
            sample_ix = np.random.choice(self.n_obs, size=self.n_obs)
            X_samples = self.X[sample_ix]
            Y_samples = self.Y[sample_ix]
            if method == 'NIPALS':
                _, XL, _, YL, B = nipals(X_samples, Y_samples, self.ncomps,
                                           n_iters=n_nipals_iters, tol=tol)
            elif method == 'SIMPLS':
                _, XL, _, YL, _, B= simpls(X_samples, Y_samples,
                                                   self.ncomps)
            elif method == 'W2A':
                _, _, XL, YL, B = pls_w2a(X_samples, Y_samples, self.ncomps)
            n = i+1
            if i==0:  
                Mx += XL
                My += YL
                Mb += B
            else:
                if i>=2:
                     c = (n-2) / (n - 1)
                     Vx =  c * Vx + (XL - Mx + (XL - Mx) / n)**2 / n
                     Vy =  c * Vy + (YL - My + (YL - My) / n)**2 / n
                     Vb =  c * Vb + (B - Mb + (B - Mb) / n)**2 / n

                Mx += (XL - Mx) / n
                My += (YL - My) / n
                Mb += (B - Mb) / n
            #Bsmp.append(B)
            self.boot_mean_LX, self.boot_mean_LY, self.boot_mean_B = Mx, My, Mb
            self.boot_var_LX, self.boot_var_LY, self.boot_var_B = Vx, Vy, Vb
            #self.Bsmp = Bsmp
            if vocal is True:
                print(i)
    def cross_validate(self, n_samples=100, method='SIMPLS', n_nipals_iters=100,
                       tol=1e-6):
        rss = 0.0
        for i in range(n_samples):
            sample_ix = np.random.choice(self.n_obs, size=int(self.n_obs/2.0))
            test_ix = [x for x in np.arange(self.n_obs) if x not in sample_ix]
            X_samples, X_test = self.X[sample_ix], self.X[test_ix]
            Y_samples, Y_test = self.Y[sample_ix], self.Y[test_ix]
            if method == 'NIPALS':
                _, XL, _, YL, B = nipals(X_samples, Y_samples, self.ncomps,
                                           n_iters=n_nipals_iters, tol=tol)
            elif method == 'SIMPLS':
                _, XL, _, YL, _, B= simpls(X_samples, Y_samples,
                                                   self.ncomps)
            elif method == 'W2A':
                _, _, XL, YL, B = pls_w2a(X_samples, Y_samples, self.ncomps)
            rss += np.mean((Y_test - X_test.dot(B))**2)/n_samples
        self.crossval_rss = rss
        

        

        
        

            
            
        
   

        
        
        
class PLS_SEM:
    
    def __init__(self, X_blocks, predictor_matrix, modes, n_iters=500, tol=1e-9,
                 vocal=False):
        self.X_blocks, self.cols, self.ixs, self.is_pds = [], [], [], []
        self.n_obs, self.block_sizes = X_blocks[0].shape[0], []
        for X in X_blocks:
            X, cols, ix, is_pd = check_type(X)
            self.X_blocks.append(X)
            self.cols.append(cols)
            self.ixs.append(ix)
            self.is_pds.append(is_pd)
            self.block_sizes.append(X.shape[1])
        
        self.n_blocks = len(self.X_blocks)
        self.predictor_matrix = predictor_matrix
        self.modes = modes
        self.n_iters = n_iters
        self.tol=tol
        self.vocal=vocal
        
        
    def weight_normalize(self, X, w):
        n= X.shape[0]
        Zh = dot(X, w)
        wn = sqrt(n)*w/norm(Zh)
        return wn
    
    def estimate_inner(self, Z, adjacency_matrix, predictor_matrix):
        p = adjacency_matrix.shape[0]
        inner_weights = []
        Zm = np.block([zk.reshape(zk.shape[0], 1) for zk in Z])
        for i in range(p):
            eq = omat(p, 1)
            if np.sum(predictor_matrix[:, i])>0:
                endog_ix = predictor_matrix[:, i].astype(bool)
                endog = Zm[:, endog_ix]
                eq[endog_ix] = lstq(endog, Zm[:, [i]])
            else:
                endog_ix = omat(p, 1).reshape(p).astype(bool)
            adj_ix = (adjacency_matrix[:, i] - endog_ix.astype(int)).astype(bool)
            eq[adj_ix]
            eq[adj_ix] = corr(Zm[:, adj_ix.astype(bool)], Zm[:, [i]])
            inner_weights.append(eq)
        inner_weights = [self.weight_normalize(Zm, inner_weights[i]) 
                         for i in range(p)]
        inner_weights = np.block(inner_weights)
        inner_approx = Zm.dot(inner_weights)
        return  inner_approx, inner_weights
                
                
    def estimate_outer(self, X_blocks, inner_approx, modes):
        n_blocks = len(X_blocks)
        W = []
        for i in range(n_blocks):
            if modes[i]=='A':
                W.append(cov(X_blocks[i], inner_approx[:, [i]]))
            elif modes[i]=='B':
                W.append(lstq(center(X_blocks[i]), center(inner_approx[:, [i]])))
        W = np.block(W)
        return W
        
        
        
    def pls_semfit(self, X_blocks, predictor_matrix, modes, n_iters=500, 
                   tol=1e-9, vocal=False):
        
        sizes = [X.shape[1] for X in X_blocks]
        adjacency_matrix = predictor_matrix+predictor_matrix.T
        W = [np.random.rand(p, 1) for p in sizes]
        W = [self.weight_normalize(X, w) for X, w in list(zip(X_blocks, W))]
        W_prev = W
        Z = [dot(X, w) for X,w in list(zip(X_blocks, W))]
        
        for i in range(n_iters):
            inner_approx, inner_weights = self.estimate_inner(Z, 
                                                              adjacency_matrix, 
                                                              predictor_matrix)
            W = self.estimate_outer(X_blocks, inner_approx, modes)
            diff = np.max(abs(np.block(W_prev)-np.block(W)))
            if diff<tol:
                break
            if vocal is not False:
                print(diff)
            W_prev = W
            Z = [dot(X, w) for X,w in list(zip(X_blocks, W.T))]
        
        Zq = [dot(X, w) for X,w in list(zip(X_blocks, W.T))]
        Xi = np.block([zk.reshape(zk.shape[0], 1) for zk in Zq])
        Beta = omat(*predictor_matrix.shape)
        for i in range(predictor_matrix.shape[0]):
            endog_ix = predictor_matrix[:, i].astype(bool)
            if np.sum(endog_ix)>0:
                endog = Xi[:, endog_ix]
                bs = lstq(endog, Xi[:, [i]])
                bs = bs.reshape(bs.shape[0])
                Beta[endog_ix, i] = bs
        return Beta, Xi, W
    
    
    def fit(self, n_iters=500, tol=1e-9, n_boot=2000, boot_iter_vocal=False,
            boot_vocal=False, vocal=False):
        Beta, W = [], []
        for i in range(n_boot):
            sample_idx = np.random.choice(self.n_obs, size=self.n_obs, 
                                          replace=True)
            X_block_sample = [X.copy()[sample_idx] for X in self.X_blocks]
            Beta_k, _, W_k = self.pls_semfit(X_block_sample, 
                                             self.predictor_matrix,
                                             self.modes, n_iters, tol,
                                             boot_iter_vocal)
            Beta_k, W_k = Beta_k[:, :, None], W_k[:, :, None]
            Beta.append(Beta_k)
            W.append(W_k)
            if boot_vocal is True:
                print(i)
        
        self.Beta_samples = np.concatenate(Beta, axis=2)
        self.W_samples = np.concatenate(W, axis=2)
        self.Beta_SE = self.Beta_samples.std(axis=2)
        self.W_SE = self.W_samples.std(axis=2)
        
        self.Beta, self.Xi, self.W = self.pls_semfit(self.X_blocks, 
                                             self.predictor_matrix,
                                             self.modes, n_iters, tol,
                                             vocal)
        loading_SE, loadings = self.W[:, [0]],  self.W_SE[:, [0]]
        
        for i in range(1, self.n_blocks):
            loadings = block_diag(loadings, self.W[:, [i]])
            loading_SE = block_diag(loading_SE, self.W_SE[:, [i]])
        
        self.W, self.W_SE = loadings, loading_SE
        self.B  = self.Beta.copy()
        
        if self.is_pds[0]:
            cols = np.concatenate([col for col in self.cols])
            lv_names = ['LV%i'%i for i in range(1, self.n_blocks+1)]
            self.W = pd.DataFrame(self.W, index=cols, columns=lv_names)
            self.W_SE = pd.DataFrame(self.W_SE, index=cols, columns=lv_names)
            self.Beta = pd.DataFrame(self.Beta, index=lv_names, columns=lv_names)
            self.Beta_SE = pd.DataFrame(self.Beta_SE, index=lv_names, columns=lv_names)
            
               
        self.Beta_t = np.divide(self.Beta, self.Beta_SE, 
                                out=np.zeros_like(self.Beta), 
                                where=self.Beta_SE!=0)
        self.W_t = np.divide(self.W, self.W_SE, 
                                out=np.zeros_like(self.W), 
                                where=self.W_SE!=0)
        
        
        if self.is_pds[0]:
            self.Beta_t = pd.DataFrame(self.Beta_t, index=lv_names, columns=lv_names)
            self.W_t = pd.DataFrame(self.W_t, index=cols, columns=lv_names)
        
        self.exog_var = np.array([np.var(X) for X in self.X_blocks])
        self.communalities = (self.W**2).sum(axis=0) / np.array(self.block_sizes)
        self.total_communality =(self.W**2).sum() / np.sum(self.block_sizes)
        self.outer_model_explained_variance = self.communalities/self.exog_var
        lv_types = (self.predictor_matrix.sum(axis=0)>0)
        self.error = self.Xi[:, lv_types] - (self.Xi.dot(self.B))[:, lv_types]
        self.sse = diag(xprod(self.error))
        self.sst = diag(xprod(self.Xi[:, lv_types]))
        self.ssr=self.sst-self.sse
        self.lv_rsquared = 1 - self.sse/self.sst
        
        if self.is_pds[0]:
            comm = self.communalities.values
        else:
            comm = self.communalities
        self.redundancy = comm[lv_types] * self.lv_rsquared
        self.total_redundancy = np.mean(self.redundancy)
        self.GoF = sqrt((self.lv_rsquared).mean()*self.communalities[lv_types].mean())
        
        self.mv_rsquared = []
        for i in range(self.n_blocks):
            Xk = self.X_blocks[i]
            Xi_k = self.Xi[:, [i]]
            Xi_k_hat = lstq_pred(Xi_k, Xk)
            er_k = Xk - Xi_k_hat
            sse = diag(xprod(er_k))
            sst = diag(xprod(Xk))
            self.mv_rsquared.append(1-sse/sst)
        self.mv_rsquared = np.concatenate(self.mv_rsquared)
            
        if self.is_pds[0]:
            self.mv_rsquared = pd.DataFrame(self.mv_rsquared, index=cols)
            
  


class sCCA:
    def __init__(self, X, Y): #analysis:ignore
        self.X, self.xcols, self.xix, self.x_is_pd = check_type(X)
        self.Y, self.ycols, self.yix, self.y_is_pd = check_type(Y)
       
        
        n, p = self.X.shape
        n, q = self.Y.shape
        wx, wy = np.ones(p), np.ones(q)
        wx /= np.linalg.norm(self.X.dot(wx))
        wy /= np.linalg.norm(self.Y.dot(wy))
        
        self.ax, self.ay = np.ones(p), np.ones(q)
        self.k = np.minimum(p, q)
        self.n, self.p, self.q = n, p, q
        self.wx, self.wy = wx, wy
        self.w = np.concatenate([wx, wy])
        self.xprod = self.X.T.dot(self.Y)
        
        
    def _get_bmat(self, X, n, p, mu): #analysis:ignore
        if p>n:
            Ip = np.eye(p)
            In = np.eye(n)
            Bx = Ip/mu - X.T.dot(np.linalg.inv(In*mu+X.dot(X.T))).dot(X) / mu
        elif p<=n:
            Ip = np.eye(p)
            Bx = np.linalg.inv(Ip*mu+X.T.dot(X))
        return Bx
        
    def _soft(self, y, tau):
        psi_y = np.sign(y) * np.maximum(np.abs(y) - tau, 0)
        return psi_y
    
    def admm(self, X=None, Y=None, n_iters=1000, tol=1e-5, vocal=True,
             lx=None, ly=None, mu=None):
        if mu is None:
            mu = 10.0
        if lx is None:
            lx = 0.5
        if ly is None:
            ly = 0.5
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        n, p, q = X.shape[0], X.shape[1], Y.shape[1]
        xprod = X.T.dot(Y)
        wx, wy = np.ones(p), np.ones(q)
        wx /= np.linalg.norm(X.dot(wx))
        wy /= np.linalg.norm(Y.dot(wy))
        
        ax, ay =  self.ax*lx, self.ay*ly
        Bx, By =  self._get_bmat(X, n, p, mu), self._get_bmat(Y, n, q, mu)
        dx, dy = wx.copy(), wy.copy()
        epsilon = np.inf
        
        m2 = mu*2.0
        for i in range(n_iters):
            vxk, vyk = self._soft(wx - dx, ax/m2), self._soft(wy - dy, ay/m2)
            wxk = Bx.dot(xprod.dot(wy)+mu*(vxk+dx))
            wyk = By.dot(xprod.T.dot(wx)+mu*(vyk+dy))
            
            wxk/=np.linalg.norm(X.dot(wxk))
            wyk/=np.linalg.norm(Y.dot(wyk))
            
            dx = dx - wxk + vxk
            dy = dy - wyk + vyk
            
            epsilon = np.linalg.norm(wxk - wx) + np.linalg.norm(wyk - wy)
            if epsilon < tol:
                break
            else:
                wx, wy = wxk, wyk
            if vocal is True:
                print(i)
        return wx, wy
    
    def fit(self, n_comps=None, lx=None, ly=None, mu=None):
        if n_comps is None:
            n_comps = self.k
        Xv, Yv = self.X.copy(), self.Y.copy()
        U, V = np.zeros((n_comps, self.p)), np.zeros((n_comps, self.q))
        for i in range(n_comps):
            if i>0:
                tx, ty = Xv.dot(wx[:, None]), Yv.dot(wy[:, None]) #analysis:ignore
                px, py = Xv.T.dot(tx)/np.dot(tx.T, tx), Yv.T.dot(ty)/np.dot(ty.T, ty)
                Xv = Xv - tx.dot(px.T)#analysis:ignore
                Yv = Yv - ty.dot(py.T)#analysis:ignore
            wx, wy = self.admm(Xv, Yv, vocal=False, lx=lx, ly=ly, mu=mu)
            U[i] = wx
            V[i] = wy
        self.U, self.V = U, V
    
    def _fit(self, X, Y, n_comps=None, lx=None, ly=None, mu=None):
        if n_comps is None:
            n_comps = self.k
        Xv, Yv = X.copy(), Y.copy()
        U, V = np.zeros((n_comps, self.p)), np.zeros((n_comps, self.q))
        for i in range(n_comps):
            if i>0:
                tx, ty = Xv.dot(wx[:, None]), Yv.dot(wy[:, None]) #analysis:ignore
                px, py = Xv.T.dot(tx)/np.dot(tx.T, tx), Yv.T.dot(ty)/np.dot(ty.T, ty)
                Xv = Xv - tx.dot(px.T)#analysis:ignore
                Yv = Yv - ty.dot(py.T)#analysis:ignore
            wx, wy = self.admm(Xv, Yv, vocal=False, lx=lx, ly=ly, mu=mu)
            U[i] = wx
            V[i] = wy
        return U, V
    
    def cross_val(self,  mu_range=None, reg_params=None, split_ratio=.7,
                  n_iters=100, n_comps=None, vocal=True):
        if mu_range is None:
            mu_range = np.arange(1.0, 20, 5)
        if reg_params is None:
            reg_params = np.arange(0.05, 2.0, 0.3)
        X, Y = self.X, self.Y
        xvals_mu = []
        m = np.round(split_ratio*self.n)
        for i in range(n_iters):
            idx = np.zeros(self.n)
            ix = np.random.choice(self.n, int(m), replace=False)
            idx[ix] = 1.0
            idx = idx.astype(bool)
            Xtr, Ytr = X[idx], Y[idx]
            Xte, Yte = X[1-idx], Y[1-idx]
            xvals_mu_i = []
            for mu_k in mu_range:
                U, V = self._fit(Xtr, Ytr, n_comps=n_comps, mu=mu_k)
                r = np.trace(np.abs(cov(Xte.dot(U.T),Yte.dot(V.T))))
                xvals_mu_i.append(r)
                if vocal is True:
                    print(i, mu_k)
            xvals_mu.append(xvals_mu_i)
        xvals_mu = np.array(xvals_mu)
        
        xvals_reg = []
        for i in range(n_iters):
            idx = np.zeros(self.n)
            ix = np.random.choice(self.n, int(m), replace=False)
            idx[ix] = 1.0
            idx = idx.astype(bool)
            Xtr, Ytr = X[idx], Y[idx]
            Xte, Yte = X[1-idx], Y[1-idx]
            xvals_reg_i = []
            for reg in reg_params:
                U, V = self._fit(Xtr, Ytr, n_comps=n_comps, lx=reg, ly=reg)
                r = np.trace(np.abs(cov(Xte.dot(U.T),Yte.dot(V.T))))
                xvals_reg_i.append(r)
                if vocal is True:
                    print(i, reg)
            xvals_reg.append(xvals_reg_i)
            
        xvals_reg = np.array(xvals_reg)
        reg = reg_params[np.argmax(xvals_reg.mean(axis=0))]
        mu = mu_range[np.argmax(xvals_mu.mean(axis=0))]
        return xvals_reg, xvals_mu, reg, mu
        
        
            
            
        
        
    
    
            
     
                  
           