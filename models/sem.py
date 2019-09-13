#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:59:05 2019

@author: lukepinkel
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from numpy import kron, eye, dot, log, trace, diag, sqrt
from scipy.stats import chi2 as chi2_dist, t as t_dist
from scipy.optimize import minimize
from numpy.linalg import inv, pinv, slogdet, det
from ..utils.base_utils import check_type, corr, cov, center
from ..utils.linalg_utils import (omat, jmat, kmat, nmat, lmat, dmat, pre_post_elim,
                            diag2, mdot, mat_rconj, lstq_pred, lstq, vec, invec,
                            vech, invech, vechc, xprod)
from ..utils.statfunc_utils import fdr_bh

class SEMModel:
    
    def __init__(self, Z, LA, BE, TH=None, PH=None, phk=2.0):
        Lmask = omat(*LA.shape)
        Ltmp = LA.copy()
        dfd = defaultdict(list) 
        for val,key in zip(*np.where(Ltmp==1)): dfd[key].append(val) 
        for key in dfd.keys():
            Lmask[dfd[key][0], key] = 1
            
        LA_idx = (Ltmp - Lmask).astype(bool)
        LA_idx = pd.DataFrame(LA_idx, index=LA.index, columns=LA.columns)
        labels = []
        if (type(LA_idx) is pd.DataFrame)|(type(LA_idx) is pd.Series):
            for x in LA_idx[LA_idx==True].stack().index.values:
                labels.append("%s ~ %s"%(x[1], x[0]))
        if (type(BE) is pd.DataFrame)|(type(BE) is pd.Series):
            for x in BE[BE==True].stack().index.values:
                labels.append("%s ~ %s"%(x[1], x[0]))   
        if PH is None:
            PH = np.eye(BE.shape[0])
        if (type(PH) is pd.DataFrame)|(type(PH) is pd.Series):
            for x in PH[PH!=0].stack().index.values:
                labels.append("r(%s ~ %s)"%(x[1], x[0]))
        else:
            tmp = pd.DataFrame(PH, index=LA.columns, columns=LA.columns)
            tix = np.triu(np.ones(tmp.shape)).astype('bool').reshape(tmp.size)
            tmp = tmp.stack()[tix]
            for x in tmp[tmp!=0].index.values:
                labels.append("resid(%s, %s)"%(x[1], x[0])) 
                
                
        if (type(TH) is pd.DataFrame)|(type(TH) is pd.Series):
            for x in TH[TH!=0].stack().index.values:
                labels.append("r(%s ~ %s)"%(x[1], x[0]))
        else:
            tmp = pd.DataFrame(TH, index=LA.index, columns=LA.index)
            tix = np.triu(np.ones(tmp.shape)).astype('bool').reshape(tmp.size)
            tmp = tmp.stack()[tix]
            for x in tmp[tmp!=0].index.values:
                labels.append("resid(%s, %s)"%(x[1], x[0]))
        self.labels=labels
        Z, self.zcols, self.zix, self.z_is_pd = check_type(Z)
        LA, self.lcols, self.lix, self.l_is_pd = check_type(LA)
        BE, self.bcols, self.bix, self.b_is_pd = check_type(BE)
        LA, idx1, BE, idx2, PH_i, idx3, TH_i, idx4 = self.init_params(Z, LA, BE, 
                                                                      TH, PH)
        if TH is None:
            TH = TH_i
        else:
            TH = TH
        PH = PH_i/phk
        p, k = LA.shape
        k1 = p * k
        k2 = k * k
        k3 = int((k + 1) * k / 2)
        k4 = int((p + 1) * p / 2)
        
        k2 = k2 + k1
        k3 =k2 + k3
        k4 = k3 + k4 
        
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.p, self.k = p, k
        self.n_obs = Z.shape[0]
        self.Z = Z
        self.S = cov(Z)
        self.LA = LA
        self.BE = BE
        self.IB = inv(mat_rconj(BE))
        self.PH = PH
        self.TH = TH
        self.idx = self.mat_to_params(idx1, idx2, idx3, idx4)
        self.params = self.mat_to_params(LA, BE, PH, TH)
        self.free = self.params[self.idx]
        self.Sigma = self.implied_cov(self.LA, self.BE, self.PH, self.TH)
        self.GLSW = pre_post_elim(kron(inv(self.S), inv(self.S)))
        self.Sinv = inv(self.S)
        self.Lp = lmat(self.p)
        self.Np = nmat(self.p)
        self.Ip = eye(self.p)
        self.Dk = dmat(self.k)
        self.Dp = dmat(self.p)
        self.Ip2 = eye(self.p**2)
        
        self.bounds = self.mat_to_params(omat(*self.LA.shape), 
                                         omat(*self.BE.shape),
                                         eye(self.PH.shape[0]),
                                         eye(self.TH.shape[0]))
        self.bounds = self.bounds[self.idx]
        self.bounds = [(None, None) if x==0 else (0, None) for x in self.bounds]
        
        
    def init_params(self, Z, L, B, TH=None, PH=None):
        BE_init = omat(*B.shape)
        BE_idx = B.copy().astype(bool)
        LA_init = omat(*L.shape)
        if TH is None:
            TH_init = diag2(cov(Z)) / 2
        else:
            TH_init = TH
        if PH is None:
            PH_init = eye(B.shape[0])*0.05
            PH_mask = eye(B.shape[0])
        else:
            PH_init = PH
            PH_mask = PH!=0
        dfd = defaultdict(list) 
        for val,key in zip(*np.where(L==1)): dfd[key].append(val) 
    
        for key in dfd.keys():
            LA_init[dfd[key][0], key] = 1
        LA_idx = (L - LA_init).astype(bool)   
        Nu = []
    
        for i in range(LA_idx.shape[1]):
            #If path model, nu, set latent var to observed var
            if LA_idx[:, i].sum()==0:
                Nu.append(Z[:, [i]])
            #Else if true structural model, use 2SLS to estimate IV model
            else:
                exog = Z[:, LA_idx[:, i]]
                endog = Z[:, LA_init[:, i].astype(bool)]
                LA_init[LA_idx[:, i], [i]] = lstq(center(exog), center(endog)).flatten()
                Nu.append(lstq_pred(center(exog), center(endog)))
            
        Nu = np.concatenate(Nu, axis=1) 
        
        for i in range(BE_idx.shape[0]):
            if np.sum(BE_idx[i])==0:
                continue
            else:
                exog = Nu[:, BE_idx[i]]
                endog = Nu[:, i]
            BE_init[i, BE_idx[i]] = lstq(center(exog), center(endog))
        PH_init = cov(Nu)*PH_mask
        PH_idx = PH_mask
        TH_idx = TH_init!=0
        return LA_init, LA_idx, BE_init, BE_idx, PH_init, PH_idx, TH_init, TH_idx

  
    def implied_cov(self, LA, BE, PH, TH):
        IB = inv(mat_rconj(BE))
        Sigma = mdot([LA, IB, PH, IB.T, LA.T]) + TH
        return Sigma
    
    def mat_to_params(self, LA, BE, PH, TH):
        params = np.block([vec(LA), vec(BE), vech(PH), vech(TH)])
        return params
  
    def get_mats(self, params=None):
        if params is None:
            params = self.params
        LA = invec(params[:self.k1], self.p, self.k)
        BE = invec(params[self.k1:self.k2], self.k, self.k)
        IB = inv(mat_rconj(BE))
        PH = invech(params[self.k2:self.k3])
        TH = invech(params[self.k3:])
        return LA, BE, IB, PH, TH
        
    def get_sigma(self, free):
        params = self.params.copy()
        if free.dtype==complex:
            params = params.astype(complex)
        params[self.idx] = free
        LA, BE, IB, PH, TH = self.get_mats(params)
        Sigma = self.implied_cov(LA, BE, PH, TH)
        return Sigma
    
    def obj_func(self, free, method='GLS'):
        if free.ndim==2:
            free = free[:, 0]
        Sigma = self.get_sigma(free)
        if method=='GLS':
            d = vech(Sigma - self.S)
            V = self.GLSW
            f = mdot([d.T, V, d])
        elif method=='ML':
            lnd = slogdet(Sigma)[1]
            f = lnd+trace(pinv(Sigma).dot(self.S))
        return f
    
    def gradient(self, free, method='GLS', full=False):
        if free.ndim==2:
            free = free[:, 0]
        params = self.params.copy()
        if free.dtype==complex:
            params = params.astype(complex)
        params[self.idx] = free
        Sigma = self.get_sigma(free)
        LA, BE, IB, PH, TH = self.get_mats(params)
        A = dot(LA, IB)
        B = mdot([A, PH, IB.T])
        
        if method=='GLS':
            DLambda = mdot([self.Lp, self.Np, kron(B, self.Ip)])
            DBeta = -mdot([self.Lp, self.Np, kron(B, A)])
            DPhi = mdot([self.Lp, kron(A, A), self.Dk])
            DPsi = mdot([self.Lp, self.Ip2, self.Dp])
            W = self.GLSW
            
            G = np.block([DLambda, DBeta, DPhi, DPsi])
            g = -2*mdot([(vechc(self.S)-vechc(Sigma)).T, W, G])
            
        elif method=='ML':
            InvSigma = pinv(Sigma)
            DLambda = vec(mdot([InvSigma, B]))-vec(mdot([InvSigma, self.S, InvSigma,
                         B]))
            DBeta = vec(mdot([A.T, InvSigma, B]))-vec(mdot([A.T, InvSigma,
                          self.S, InvSigma, B]))
            DPhi = vech(mdot([A.T, InvSigma, A])) - vech(mdot([A.T, InvSigma,
                       self.S, InvSigma, A]))
            DPsi = vech(InvSigma) - vech(mdot([InvSigma, self.S, InvSigma]))
            
            g = np.concatenate([DLambda, DBeta, DPhi, DPsi])
            g = g[:, None].T
        if full==False:
            grad = g[0][self.idx]
        else:
            grad = g[0]
        return grad
    
    def dsigma(self, free):
        params = self.params.copy()
        params[self.idx] = free
        LA, BE, IB, PH, TH = self.get_mats(params)
        A = dot(LA, IB)
        B = mdot([A, PH, IB.T])
        DLambda = mdot([self.Lp, self.Np, kron(B, self.Ip)])
        DBeta = mdot([self.Lp, self.Np, kron(B, A)])
        DPhi = mdot([self.Lp, kron(A, A), self.Dk])
        DPsi = mdot([self.Lp, self.Ip2, self.Dp])        
        G = np.block([DLambda, DBeta, DPhi, DPsi])
        return G
    
    def hessian(self, free, method='GLS'):
        params = self.params.copy()
        params[self.idx] = free
        Sigma = self.get_sigma(free)
        LA, BE, IB, PH, TH = self.get_mats(params)
        A = dot(LA, IB)
        B = mdot([A, PH, IB.T])
        if method=='GLS':
            DLambda = mdot([self.Lp, self.Np, kron(B, self.Ip)])
            DBeta = -mdot([self.Lp, self.Np, kron(B, A)])
            DPhi = mdot([self.Lp, kron(A, A), self.Dk])
            DPsi = mdot([self.Lp, self.Ip2, self.Dp])
            W = self.GLSW
         
            G = np.block([DLambda, DBeta, DPhi, DPsi])
            Sinv = pinv(self.S)
            InvSigma = pinv(Sigma)
            W = kron(Sinv, mdot([Sinv, self.S-Sigma, Sinv]))
            W = pre_post_elim(W)
            H = -2*mdot([G.T, W, G])
        elif method=='ML':
            DLambda = mdot([self.Lp, self.Np, kron(B, self.Ip)])
            DBeta = -mdot([self.Lp, self.Np, kron(B, A)])
            DPhi = mdot([self.Lp, kron(A, A), self.Dk])
            DPsi = mdot([self.Lp, self.Ip2, self.Dp])
            G = np.block([DLambda, DBeta, DPhi, DPsi])
            Sinv = pinv(self.S)
            InvSigma = pinv(Sigma)
            ESE = mdot([InvSigma, self.S, InvSigma])
            W = kron(InvSigma, ESE) + kron(ESE, InvSigma) - kron(InvSigma, InvSigma)
            W = pre_post_elim(W)
            H = -mdot([G.T, W, G])
        return H[self.idx][:, self.idx]

    def einfo(self, free):
        params = self.params.copy()
        params[self.idx] = free
        Sigma = self.get_sigma(free)
        Sinv = inv(Sigma)
        D = dmat(Sinv.shape[0])
        W = 2*mdot([D.T, kron(Sinv, Sinv), D])
        G = self.dsigma(free)[:, self.idx]
        ncov = pinv(mdot([G.T, W, G]))
        return ncov
    
    def fit(self, method='ML', xtol=1e-20, gtol=1e-30, maxiter=3000, verbose=2):
        self.optimizer = minimize(self.obj_func, self.free, 
                                  args=('ML',), jac=self.gradient,
                                  hess=self.hessian, method='trust-constr',
                                  bounds=self.bounds, options={'xtol':xtol, 
                                                               'gtol':gtol,
                                                               'maxiter':maxiter,
                                                               'verbose':verbose})    
        params = self.params.copy()
        params[self.idx] = self.optimizer.x           
        self.LA, self.BE, self.IB, self.PH, self.TH = self.get_mats(params)      
        self.free = self.optimizer.x      
        self.Sigma = self.get_sigma(self.free)
        Sinv = pinv(self.Sigma)
        W = pre_post_elim(kron(Sinv, Sinv))+np.outer(vech(Sinv), vech(Sinv))
        Delta = self.dsigma(self.free)
        W = mdot([Delta.T, W, Delta])[self.idx][:, self.idx]
        self.SE_exp = 2*diag(self.einfo(self.free)/self.n_obs)**0.5
        self.SE_obs = diag(inv(-self.hessian(self.free, 'ML'))/self.n_obs)**0.5
        self.SE_rob = diag(pinv(W) / self.n_obs)**0.5
        self.res = pd.DataFrame([self.free, self.SE_exp, self.SE_obs, self.SE_rob], 
                                index=['Coefs','SE1', 'SE2', 'SEr'], 
                                columns=self.labels).T
        self.test_stat = (self.n_obs-1)*(self.obj_func(self.free, 'ML')\
                         - slogdet(self.S)[1]-self.S.shape[0])
        self.df = len(vech(self.S))-len(self.free)
        self.test_pval = 1.0 - chi2_dist.cdf(self.test_stat, self.df)
        self.res['t'] = self.res['Coefs'] / self.res['SE1']
        self.res['p'] = 1 - t_dist.cdf(abs(self.res['t']), self.n_obs)
        self.res['adj p'] = fdr_bh(self.res['p'])
        p  = self.S.shape[0]
        normalized_resids = 0.0
        SMSR = 0.0
        s, sig = check_type(self.S)[0], check_type(self.Sigma)[0]
        for i in range(self.S.shape[0]):
            for j in range(self.S.shape[0]):
                if i<=j:
                    sigij, sigjj, sigii = sig[i,j], sig[j, j], sig[i, i]
                    nrij = (s[i, j]-sigij)/sqrt((sigjj*sigii+sigij**2)/self.n_obs)
                    normalized_resids+= nrij
                    SMSR += (s[i, j] - sigij)**2 / (s[i, i]*s[j, j])
        self.SMSR = sqrt(2*SMSR/(p*(p+1)))
        self.normalized_resids = normalized_resids
        V = pinv(self.Sigma)
        Ip = eye(self.S.shape[0])
        self.GFI = trace(xprod(dot(V, self.S)-Ip))/trace(xprod(dot(V, self.S)))
        if self.df!=0:
            self.AGFI = 1 - (p*(p+1)/self.df)*(1-self.GFI)
            self.st_chi2 = (self.test_stat - self.df) / sqrt(2*self.df)
            self.RMSEA = sqrt(np.maximum(self.test_stat-self.df, 
                                     0)/(self.df*self.n_obs-1))      