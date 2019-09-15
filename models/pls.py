#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:55:37 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
from numpy import dot, sqrt, eye, diag, log, zeros
from numpy.linalg import multi_dot, norm, pinv
from scipy.stats import f as f_dist, chi2 as chi2_dist, t as t_dist
from scipy.stats import scoreatpercentile
from scipy.linalg import block_diag

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
        
    def fit(self):
        
        X_loadings, singular_values, Y_loadings = svd2(self.R)
        X_var, Y_var = dot(self.X, X_loadings), dot(self.Y, Y_loadings)
        
        
        
        self.singular_values = singular_values
        
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
            
    def permutation_test(self):
        svs = []
        for i in range(1000):
            Rperm = corr(np.random.permutation(self.X.copy()), self.Y)
            U, inertia, V = svd2(Rperm)
            svs.append(inertia[:, None])
        self.singular_value_dist = np.concatenate(svs, axis=2)
        
    
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
        
 
def simpls(X, Y, ncomps):
    S = dot(X.T, Y)
    n, m, q = X.shape[0], X.shape[1], Y.shape[1]
    T, P = np.zeros((n, ncomps)), np.zeros((m, ncomps))
    U, Q = np.zeros((n, ncomps)), np.zeros((q, ncomps))
    V = np.zeros((m, ncomps))
    R = np.zeros((m, ncomps))
    
    for i in range(ncomps):
        eigvals, eigvecs = sorted_eigh(dot(S.T, S))
        r = dot(S, eigvecs[:, [0]])
        t = dot(X, r)
        t -= np.mean(t)
        r /= norm(t)
        t /= norm(t)
        p = dot(X.T, t)
        q = dot(Y.T, t)
        u = dot(Y, q)
        v = p
        if i>1:
            v = v - multi_dot([V, V.T, p])
            u = u - multi_dot([T, T.T, u])
        v /= sqrt(dot(v.T, v))
        S -= multi_dot([v, v.T, S])
        
        R[:, i] = r[:, 0]
        T[:, i] = t[:, 0]
        P[:, i] = p[:, 0]
        Q[:, i] = q[:, 0]
        U[:, i] = u[:, 0]
        V[:, i] = v[:, 0]
        
    B  = dot(R, Q.T)
    U /= sqrt(norm(U, axis=0))
    Q /= norm(Q, axis=0)
    Vp = norm(P, axis=0)
    P /= Vp
    T *= Vp
    return T, P, U, Q, R, V, B



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
        
        if method is 'NIPALS':
            self.x_factors, self.x_loadings, self.y_factors, \
            self.y_loadings, self.coefs = nipals(self.X, self.Y, self.ncomps,
                                                 n_iters=n_iters, tol=tol)
        elif method is 'SIMPLS':
            self.x_factors, self.x_loadings, self.y_factors, \
            self.y_loadings, self.x_weights, self.y_weights, \
            self.coefs = simpls(self.X, self.Y, self.ncomps)
            
        elif method is 'W2A':
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
        x_loading_samples = []
        y_loading_samples = []
        coef_samples = []
        for i in range(n_samples):
            sample_ix = np.random.choice(self.n_obs, size=self.n_obs)
            X_samples = self.X[sample_ix]
            Y_samples = self.Y[sample_ix]
            if method is 'NIPALS':
                _, XL, _, YL, B = nipals(X_samples, Y_samples, self.ncomps,
                                           n_iters=n_nipals_iters, tol=tol)
            elif method is 'SIMPLS':
                _, XL, _, YL, XW, YW, B = simpls(X_samples, Y_samples,
                                                   self.ncomps)
            elif method is 'W2A':
                _, _, XL, YL, B = pls_w2a(X_samples, Y_samples, self.ncomps)
                
            x_loading_samples.append(XL[:, :, np.newaxis])
            y_loading_samples.append(YL[:, :, np.newaxis])
            coef_samples.append(B[:, :, np.newaxis])
            if vocal is True:
                print(i)
        
        self.xl_samples = np.concatenate(x_loading_samples, axis=2)
        self.yl_samples = np.concatenate(y_loading_samples, axis=2)
        self.coef_samples = np.concatenate(coef_samples, axis=2)
        
        self.xl_mean = np.mean(self.xl_samples, axis=2)
        self.yl_mean = np.mean(self.yl_samples, axis=2)
        self.coef_mean = np.mean(self.coef_samples, axis=2)
        
        self.xl_confint_up = scoreatpercentile(self.xl_samples, CI, axis=2)
        self.yl_confint_up = scoreatpercentile(self.yl_samples, CI, axis=2)
        self.coef_confint_up = scoreatpercentile(self.coef_samples, CI, axis=2)
    
        self.xl_confint_low = scoreatpercentile(self.xl_samples,
                                                100-CI, axis=2)
        
        self.yl_confint_low = scoreatpercentile(self.yl_samples,
                                                100-CI, axis=2)
        
        self.coef_confint_low = scoreatpercentile(self.coef_samples,
                                                  100.0-CI, axis=2)        
        self.boostrap_tcrit = t_dist.ppf(CI/100.0, n_samples-2)
        self.xl_se = np.std(self.xl_samples, axis=2)
        self.yl_se = np.std(self.yl_samples, axis=2)
        self.coef_se=np.std(self.coef_samples, axis=2)
        
        self.xl_tvalue = self.xl_mean / self.xl_se
        self.yl_tvalue = self.yl_mean / self.yl_se
        self.coef_tvalue = self.coef_mean / self.coef_se 
        
        if self.x_is_pd or self.y_is_pd:
            self.coef_tvalue = pd.DataFrame(self.coef_tvalue,
                                            index=self.xcols, 
                                            columns=self.ycols)
            
            
   

        
        
        
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
            
            
           