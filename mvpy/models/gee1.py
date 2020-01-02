#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 18:43:40 2019

@author: lukepinkel
"""

import patsy 
import collections 
import numpy as np 
import scipy as sp 
import scipy.stats
import pandas as pd 
from .utils import linalg_utils
from .glm3 import Binomial, LogitLink, GLM
def sherman_morrison(a, ni):
    t = 1.0 / a + ni / (1 - a)
    u = 1 - a
    usq = u**2
    aii = 1 / u - 1 / (t * usq)
    aij = - 1 / (t * usq)
    return aii, aij
    

class Exchangeable:
    
    def __init__(self):
        self.alpha = 0.0
        self.npars=1
    
    def init_cov(self, cix, clen):
        self.cix, self.clen = cix, clen
        self.mix = {}
        self.mats = {}
        self.rxpx = {}
        self.n = 0.0
        self.dmats = {}
        self.drda = {}
        self.sizes = {}
        self.unique_sizes = np.unique(list(self.clen.values()))
        for i in self.cix:
            ni = int(self.clen[i])
            self.mix[i] = np.where((np.ones((ni, ni)) - np.eye(ni))==1)
            self.mats[i] = np.eye(ni)
            self.n += ni * (ni - 1)
            self.rxpx[i] = np.tril_indices(ni, -1)
            self.sizes[i] = ni
        for x in self.unique_sizes:
            self.dmats[x] = np.linalg.pinv(linalg_utils.dmat(x))
            self.drda[x] = linalg_utils.vech(np.ones((ni, ni)) - np.eye(ni))
            
        self.n *= 0.5

    def estimate(self, residuals, scale, p):
        alpha = 0.0
        for i in self.cix:
            ri = residuals[i]
            ix1, ix2 = self.rxpx[i]
            alpha+= np.sum(ri[ix1]*ri[ix2])
        alpha = alpha / ((self.n - p) * scale)
        return alpha    
    
    def unpack(self, params):
        beta, alpha = params[:-1], params[-1:]
        return beta, alpha
            
    def get_corr(self, alpha, i):
        ix = self.mix[i]
        Ri = self.mats[i].copy()
        Ri[ix] = alpha
        return Ri
    
    def get_invcorr(self, alpha, i):
        ix = self.mix[i]
        Ri = self.mats[i].copy()
        ni = Ri.shape[0]
        aii, aij = sherman_morrison(alpha, ni)
        Ri[np.diag_indices(ni)] = aii
        Ri[ix] = aij
        return Ri
    
    def get_cov(self, alpha, i, vsq):
        Ri = self.get_corr(alpha, i)
        Vi = vsq * Ri * vsq.T
        return Vi
    
    def get_invcov(self, alpha, i, vsq):
        visq = 1.0 / vsq
        Rinvi = self.get_invcorr(alpha, i)
        Vinvi = visq * Rinvi * visq.T
        return Vinvi
    
    def dcorr(self, i, alpha):
        dR = self.drda[self.clen[i]]
        return dR
    
    def get_dmat(self, i):
        Dp = self.dmats[self.clen[i]]
        return Dp
        
        
        
    
class Independent:
    
    def __init__(self):
         self.alpha = 0.0
         self.npars = 1.0
         
    def init_cov(self, cix, clen):
        self.cix, self.clen = cix, clen
        self.mix = {}
        self.mats = {}
        self.unique_sizes = np.unique(list(self.clen.values()))
        for i in self.cix:
            ni = int(self.clen[i])
            self.mats[i] = np.eye(ni)
        for x in self.unique_sizes:
            self.dmats[x] = np.linalg.pinv(linalg_utils.dmat(x))
            self.drda[x] = linalg_utils.vech(np.zeros((ni, ni)))
            
    def unpack(self, params):
        return params, None
    
    def estimate(self, residuals, scale, p):
        alpha = 0.0
        return alpha
    
    def get_cov(self, alpha, i, vsq):
        Ri = self.get_corr(alpha, i)
        Vi = vsq * Ri * vsq.T
        return Vi
    
    def get_corr(self, alpha, i):
        return self.mats[i]
    
    def get_invcorr(self, alpha, i):
        return self.mats[i]
    
    def get_invcov(self, alpha, i, vsq):
        visq = 1.0 / vsq
        Rinvi = self.mats[i]
        Vinvi = visq * Rinvi * visq.T
        return Vinvi
    
    def dcorr(self, i, alpha):
        return self.drda[i]*0.0
    
    def get_dmat(self, i):
        Dp = self.dmats[self.clen[i]]
        return Dp
        
        
    
class MDependent:
    
    def __init__(self, m=1):
           self.alpha = 0.0
           self.m =  m
           self.npars = m
           
    def init_cov(self, cix, clen):
        self.cix, self.clen = cix, clen
        self.mix = {}
        self.dmats = {}
        self.drda = {}
        self.mats = {}
        m = self.m
        self.n = 0.0
        self.unique_sizes = np.unique(list(self.clen.values()))
        for i in self.cix:
            ni = int(self.clen[i])
            rows, cols = np.indices((ni, ni))
            ixa, ixb = [], []
            for j in range(1, m+1):
                ixa_j, ixb_j = np.diag(rows, k=-j),  np.diag(cols, k=-j)
                ixa.append(ixa_j)
                ixb.append(ixb_j)
            self.mix[i] = list(zip(ixa, ixb))
            self.mats[i] = np.eye(ni)
            self.n += ni
        for x in self.unique_sizes:
            self.dmats[x] = np.linalg.pinv(linalg_utils.dmat(x))
            rows, cols = np.indices((x, x))
            ixa, ixb = [], []
            for j in range(1, m+1):
                ixa_j, ixb_j = np.diag(rows, k=-j),  np.diag(cols, k=-j)
                ixa.append(ixa_j)
                ixb.append(ixb_j)
            
            ix = list(zip(ixa, ixb))
            D = []
            for z in ix:
                A = np.zeros((ni, ni))
                A[z] = 1
                A[z[::-1]] = 1
                D.append(linalg_utils.vechc(A))
            D = np.concatenate(D, axis=1)
            self.drda[x] = D
            
    def estimate(self, residuals, scale, p):
        alpha = np.zeros(self.m)
        k = len(self.cix)
        for j in range(self.m):
            kt = self.n - k * j
            for i in self.cix:
                ix = self.mix[i]
                ri = residuals[i]
                alpha[j] += np.sum(ri[ix[j][0]] * ri[ix[j][1]])
            alpha[j] /= ((kt - p) * scale)
        return alpha
           

    def unpack(self, params):
        beta, alpha = params[:-self.m], params[-self.m:]
        return beta, alpha
   
    
    def get_corr(self, alpha, i):
        ix = self.mix[i]
        Ri = self.mats[i].copy()
        for j, x in enumerate(ix):
            Ri[x] = alpha[j]
            Ri[x[::-1]] = alpha[j]
        #Ri[ix] = alpha
        return Ri
    
    def get_invcorr(self, alpha, i):
        return np.linalg.inv(self.get_corr(alpha, i))
    
    def dcorr(self, i, alpha):
        return self.drda[self.clen[i]]
    
             
    def get_cov(self, alpha, i, vsq):
        Ri = self.get_corr(alpha, i)
        Vi = vsq * Ri * vsq.T
        return Vi
    
    def get_invcov(self, alpha, i, vsq):
        visq = 1.0 / vsq
        Rinvi = np.linalg.inv(self.get_corr(alpha, i))
        Vinvi = visq * Rinvi * visq.T
        return Vinvi
    
    def get_dmat(self, i):
        Dp = self.dmats[self.clen[i]]
        return Dp
        
        

class Unstructured:
    
    def __init__(self):
        self.alpha = 0.0
        
    def init_cov(self, cix, clen):
        self.cix, self.clen = cix, clen
        self.mix = {}
        self.mats = {}
        self.n = 00
        for i in self.cix:
            ni = int(self.clen[i])
            self.mix[i] = np.triu_indices(ni, 1)
            self.mats[i] = np.eye(ni)
            self.n+=ni
        self.npars=ni
        self.unique_sizes = np.unique(list(self.clen.values()))     
        for x in self.unique_sizes:
            self.dmats[x] = np.linalg.pinv(linalg_utils.dmat(x))
            
    
    def estimate(self, residuals, scale, p):
        alpha = np.zeros(len(self.mix[0]))
        for i in self.cix:
            ri = residuals[i]
            ixa, ixb = self.mix[i]
            alpha+= ri[ixa]*ri[ixb]
        alpha/=((self.n - p)*self.scale)
        return alpha
        
            
    def get_corr(self, alpha, i):
        ix = self.mix[i]
        Ri = self.mats[i].copy()
        Ri[ix] = alpha
        Ri[ix[::-1]] = alpha
        return Ri
    
    def get_invcorr(self, alpha, i):
        return np.linalg.inv(self.get_corr(alpha, i))
    
    def get_cov(self, alpha, i, vsq):
        Ri = self.get_corr(alpha, i)
        Vi = vsq * Ri * vsq.T
        return Vi
    
    def get_invcov(self, alpha, i, vsq):
        visq = 1.0 / vsq
        Rinvi = np.linalg.inv(self.get_corr(alpha, i))
        Vinvi = visq * Rinvi * visq.T
        return Vinvi
        
    def get_dmat(self, i):
        Dp = self.dmats[self.clen[i]]
        return Dp
    
    
        
        
        
        
class AR1:
    
    def __init__(self):
        self.alpha = 0.0
        self.npars=1
        
    def init_cov(self, cix, clen):
        self.cix, self.clen = cix, clen
        self.mix = {}
        self.mats = {}
        self.dmats = {}
        self.drda = {}
        self.mixinv = {}
        self.n = 0.0
        self.unique_sizes = np.unique(list(self.clen.values()))
        for i in self.cix:
            ni = int(self.clen[i])
            self.mats[i] = sp.linalg.toeplitz(np.arange(ni).astype(float))
            rows, cols = np.indices((ni, ni))
            ixa, ixb = [], []
            for j in range(1, 2):
                ixa_j, ixb_j = np.diag(rows, k=-j),  np.diag(cols, k=-j)
                ixa.append(ixa_j)
                ixb.append(ixb_j)
            self.n += ni - 1.0
            ixa, ixb = np.concatenate(ixa), np.concatenate(ixb)
            self.mix[i] = np.tril_indices(ni, -1)
            self.mixinv[i] = (np.concatenate([ixa, ixb]), np.concatenate([ixb, ixa]))
        
        for x in self.unique_sizes:
            self.dmats[x] = np.linalg.pinv(linalg_utils.dmat(x))
            
    def estimate(self, residuals, scale, p):
        alpha = 0.05
        for i in self.cix:
            ri = residuals[i]
            ixa, ixb = self.mix[i]
            alpha+= np.sum(ri[ixa]*ri[ixb])
        alpha/=((self.n - p)*scale)
        return alpha

    def unpack(self, params):
        beta, alpha = params[:-1], params[-1:]
        return beta, alpha
    
    def get_corr(self, alpha, i):
        Ri = self.mats[i].copy()
        Ri = alpha**Ri
        return Ri
    
    def get_invcorr(self, alpha, i):
        ix = self.mixinv[i]
        Ri = np.eye(self.mats[i].copy().shape[0])
        ni = Ri.shape[0]
        Ri[ix] = -alpha
        #Ri[ix[1], ix[0]] = -alpha
        Ri[np.arange(1, ni-1), np.arange(1, ni-1)] = 1 + alpha**2
        Ri[0, 0] = Ri[-1, -1] = 1.0
        Ri/= (1 - alpha**2)
        return Ri
    
    def get_cov(self, alpha, i, vsq):
        Ri = self.get_corr(alpha, i)
        Vi = vsq * Ri * vsq.T
        return Vi
    
    def get_invcov(self, alpha, i, vsq):
        visq = 1.0 / vsq
        Rinvi = self.get_invcorr(alpha, i)
        Vinvi = visq * Rinvi * visq.T
        return Vinvi
    
    def dcorr(self, i, alpha):
        return linalg_utils.vech(self.mats[i]*(alpha)**(self.mats[i]-1))
    
    def get_dmat(self, i):
        Dp = self.dmats[self.clen[i]]
        return Dp
    
    


class GEE:
    
    def __init__(self, frm, grps, data, fm=Binomial(LogitLink), wcov=Exchangeable):
        Y, X = patsy.dmatrices(frm, data, return_type='dataframe')
        z = data[grps]
        self.xnames = X.columns
        self.Xgrouper = X.groupby(z)
        self.Ygrouper = Y.groupby(z)
        self.gdic = self.Xgrouper.groups
        self.cix = list(self.gdic.keys())
        self.Xg = collections.defaultdict()
        self.Yg = collections.defaultdict()
        self.clen = collections.defaultdict()
        self.X = linalg_utils._check_np(X)
        self.Y = linalg_utils._check_np(Y)
        for i in self.cix:
            self.Xg[i] = self.Xgrouper.get_group(i).values
            self.Yg[i] = self.Ygrouper.get_group(i).values
            self.clen[i] = len(self.Xg[i])
        try:
            self.wcov = wcov()
        except TypeError:
            self.wcov = wcov
        self.wcov.init_cov(self.cix, self.clen)
        self.f = fm
        self.n, self.p = X.shape
        self.frm, self.data = frm, data
    
    def _predict(self, params, i):
        beta, alpha = self.wcov.unpack(params)
        Xi = self.Xg[i]
        eta = Xi.dot(beta)
        mu = self.f.inv_link(eta)
        return mu
    
    def predict(self, params):
        mu_dict = {}
        for i in self.cix:
            mu_dict[i] = self._predict(params, i)
        return mu_dict
    
    def get_resids(self, params):
        mu = self.predict(params)
        T, sd, resids = {}, {}, {}
        for i in self.cix:
            T[i] = self.f.canonical_parameter(mu[i])
            sd[i] = linalg_utils._check_2d(np.sqrt(self.f.var_func(T[i])))
            resids[i] = self.Yg[i] - linalg_utils._check_2d(mu[i])
        return resids, sd
     
    def get_presids(self, params):
        mu = self.predict(params)
        T, presids = {}, {}
        for i in self.cix:
            T[i] = self.f.canonical_parameter(mu[i])
            sdi = linalg_utils._check_2d(np.sqrt(self.f.var_func(T[i])))
            presids[i] = (self.Yg[i] - linalg_utils._check_2d(mu[i]))/sdi
        return presids 
            
    def _s_iter(self, params, scale, i):
        beta, alpha = self.wcov.unpack(params)
        Xi = self.Xg[i]
        Yi = linalg_utils._check_2d(self.Yg[i])
        eta = Xi.dot(beta)
        mu = self.f.inv_link(eta)
        ri = Yi - linalg_utils._check_2d(mu)
        T = self.f.canonical_parameter(mu)
        vsq = linalg_utils._check_2d(np.sqrt(self.f.var_func(T)))
        D = Xi * linalg_utils._check_2d(self.f.dinv_link(eta))
        Vi = self.wcov.get_invcov(alpha, i, vsq) / scale
        Hi = np.matmul(D.T, Vi).dot(D)
        Ui = np.matmul(D.T, Vi).dot(ri)
        return Ui, Hi, ri
    
    def s_iter(self, params, scale):
        g = np.zeros((self.p, 1))
        H = np.zeros((self.p, self.p))
        resid_dict = collections.defaultdict()
        for i in self.cix:
            gi, Hi, ri = self._s_iter(params, scale, i)
            g+=gi
            H+=Hi
            resid_dict[i] = ri
        return H, g, resid_dict
    
    def est_scale(self, presids):
        s = 0.0
        for i in self.cix:
            s+= np.sum(presids[i]**2)
        s /= (self.n - self.p)
        return s
    
    def robust_cov(self, params, scale):
        A = 0.0
        B = 0.0
        for i in self.cix:
            beta, alpha = self.wcov.unpack(params)
            Xi = self.Xg[i]
            Yi = linalg_utils._check_2d(self.Yg[i])
            eta = Xi.dot(beta)
            mu = self.f.inv_link(eta)
            ri = Yi - linalg_utils._check_2d(mu)
            T = self.f.canonical_parameter(mu)
            vsq = linalg_utils._check_2d(np.sqrt(self.f.var_func(T)))
            D = Xi * linalg_utils._check_2d(self.f.dinv_link(eta))
            Vi = self.wcov.get_invcov(alpha, i, vsq) / scale
            Hi = np.matmul(D.T, Vi).dot(D)
            A+=Hi
            B+= np.linalg.multi_dot([D.T, Vi, np.dot(ri, ri.T), Vi, D])
        V = np.linalg.inv(A).dot(B).dot(np.linalg.inv(A))
        return V
    
    
    def varscore(self, params, vocal=False):
        beta, alpha = self.wcov.unpack(params)
        scale = self.est_scale(self.get_presids(params))
        U = 0.0
        H = 0.0
        for i in self.cix:
            Xi, Yi = self.Xg[i], self.Yg[i]
            eta = Xi.dot(beta)
            mu = self.f.inv_link(eta)
            ri = Yi - linalg_utils._check_2d(mu)
            T = self.f.canonical_parameter(mu)
            vsq = linalg_utils._check_2d(np.sqrt(self.f.var_func(T)))
            Dp = self.wcov.get_dmat(i)
            V = self.wcov.get_cov(alpha, i, vsq)  * scale
            W = self.wcov.get_invcov(alpha, i, vsq) / scale
            WW = np.kron(W, W)
            G = Dp.dot(WW).dot(Dp.T)
            drda = self.wcov.dcorr(i, alpha)
            r = linalg_utils.vech(V) - linalg_utils.vech(np.outer(ri, ri))
            d = (linalg_utils.vech(np.outer(vsq, vsq))*drda)
            C = d.dot(G)
            U+= C.dot(r)
            H = np.dot(C.T, C)
            if vocal:
                print(i)
            
        return U/len(self.cix), np.sqrt(1/H)
            
    def fit(self, n_iters=50, tol=1e-6, vocal=True):
        self.glm = GLM(self.frm, self.data, self.f)
        self.glm.fit()
        scale = np.sum(self.glm.pearson_resid**2)/(self.n-self.p)
        npars = self.wcov.npars
        params = np.concatenate([self.glm.beta, np.zeros(npars)])
        p = self.p
        self.fit_hist = {}
        for i in range(n_iters):
            H, g, resids = self.s_iter(params, scale)
            beta = params[:-npars] + linalg_utils._check_1d(np.linalg.inv(H).dot(g))
            beta = linalg_utils._check_1d(beta)
            diff = linalg_utils.normdiff(params[:-npars], beta)
            if diff<tol:
                break
            params[:-npars] = beta
            presids = self.get_presids(params)
            scale = self.est_scale(presids)
            params[-self.wcov.npars:] = self.wcov.estimate(presids, scale, p)
            if vocal:
                print(i, params, scale)
            self.fit_hist[i] = {'H':H, 'g':g, 'diff':diff, 'params':params,
                                 'scale':scale}
        self.params = params
        self.beta = params[:-npars]
        self.alpha = params[-npars:]
        self.scale=scale
        self.vcov = self.robust_cov(self.params, self.scale)
        self.SE_params = np.sqrt(np.diag(self.vcov))
        tmp = self.params[:-self.wcov.npars][:, None]
        tmp = np.concatenate([tmp, self.SE_params[:, None]], axis=1)
        self.res = pd.DataFrame(tmp, columns=['param', 'SE'],
                                index=self.xnames)
        self.res['t'] = self.res['param'] / self.res['SE']
        self.res['p'] = sp.stats.t.sf(np.abs(self.res['t']),
                                      self.n-self.p)*2.0
        
    
        
     
