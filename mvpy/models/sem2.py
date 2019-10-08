#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:37:18 2019

@author: lukepinkel
"""
import pandas as pd#analysis:ignore
import numpy as np #analysis:ignore
import scipy as sp #analysis:ignore
import scipy.stats #analysis:ignore
import collections#analysis:ignore
from ..utils import linalg_utils, base_utils, statfunc_utils #analysis:ignore


class ObjFuncQD:
    
    def __init__(self, W=None, Winv=None, S=None, V=None):
        if Winv is None:
            Winv = np.linalg.pinv(W)
        self.S = linalg_utils._check_np(S)
        self.Winv = Winv
        self.D = linalg_utils.dmat(Winv.shape[0])
        self.Dp = np.linalg.pinv(self.D)
        if V is None:
            self.V = 0.5 * self.D.T.dot(np.kron(self.Winv, self.Winv)).dot(self.D)
        else:
            self.V = V
            
    def func(self, Sigma):
        d = linalg_utils.vechc(self.S - Sigma)
        f = linalg_utils._check_0d(d.T.dot(self.V).dot(d))
        return f
    
    __call__ = func
    
    def gradient(self, Sigma, G):
        d = linalg_utils.vechc(self.S - Sigma)
        g = -2.0*G.T.dot(self.V).dot(d)
        return g
    
    def hessian(self, Sigma, G):
        H = -2* G.T.dot(self.V).dot(G)
        return H
    
    


class ObjFuncTR:
    
    def __init__(self, W=None, Winv=None, S=None):
        if Winv is None:
            Winv = np.linalg.pinv(W)
        self.S = linalg_utils._check_np(S)
        self.Winv = Winv
        self.D = linalg_utils.dmat(Winv.shape[0])
        self.Dp = np.linalg.pinv(self.D)
        self.V = 0.5 * self.D.T.dot(np.kron(self.Winv, self.Winv)).dot(self.D)
        
    def func(self, Sigma):
        D = self.S - Sigma
        f = 0.5*np.trace(self.Winv.dot(D).dot(self.Winv).dot(D))
        f = linalg_utils._check_0d(f)
        return f
    __call__ = func
    
    def gradient(self, Sigma, G):
        d = linalg_utils.vechc(self.S - Sigma)
        g = -2.0*G.T.dot(self.V).dot(d)
        return g
    
    def hessian(self, Sigma, G):
        H = -2* G.T.dot(self.V).dot(G)
        return H
    
    
class ObjFuncML:
    def __init__(self, W=None, Winv=None, S=None):
        self.W = linalg_utils._check_np(W)
        self.D = linalg_utils.dmat(W.shape[0])
        
    def func(self, Sigma=None, Sigma_inv=None):
        if Sigma_inv is None:
            Sigma_inv = np.linalg.pinv(Sigma)
        lnd = np.linalg.slogdet(Sigma)[1]
        trWS = np.trace(self.W.dot(Sigma_inv))
        f = linalg_utils._check_0d(lnd+trWS)
        return f
    
    __call__ = func

    def gradient(self, Sigma, G):
        d = linalg_utils.vechc(self.W - Sigma)
        Sigma_inv = np.linalg.pinv(Sigma)
        V = self.D.T.dot(np.kron(Sigma_inv, Sigma_inv)).dot(self.D)
        g = -G.T.dot(V).dot(d)
        return g
    
    def hessian(self, Sigma, G):
        Sigma_inv = np.linalg.pinv(Sigma)
        ESE = Sigma_inv.dot(self.W - Sigma).dot(Sigma_inv)
        V = np.kron(ESE, Sigma_inv) + np.kron(Sigma_inv, ESE)\
            -np.kron(Sigma_inv, Sigma_inv)
        V = self.D.T.dot(V).dot(self.D)
        
        H = 2*G.T.dot(V).dot(G)
        return H




class SEM:
    """
    Structural Equation Model
    
  
    
    The class is initialized with 2 Stage Least Squares parameter estimates,
    as the newton-raphson optimization is quite sensitive to starting values.
    
    Parameters
    ----------
    Z : DataFrame
        Pandas DataFrame whose rows and columns correspond to observations
        and variables, respectively.
    LA : DataFrame
        Lambda, the loadings matrix that specifies which variables load onto
        the latent variables.  For a path model(i.e. no measurement model)
        this can just be the identity matrix
    BE: DataFrame
        Beta, the matrix that specifies the structural relationships.  
        Due to a suboptimality somewhere in the code, this does not exactly
        reflect the matrix one would expect based off of a generative model,
        and so each (i, j) element, which may be either boolean(True or False)
        or binary (1, 0), specifies that variable i is explaining some variance
        in variable j(i.e. i-->j)
    TH: DataFrame
        Theta, the measurement model error covariance matrix, analogous
        to the uniqueness in factor analysis, except orthogonality is not
        necessary
    PH: DataFrame
        Phi, the latent variable covariance matrix,  
    phk: numeric
        Factor by which to divide the 2SLS estimate of Phi by
    """
    
    def __init__(self, Z, LA, BE, TH=None, PH=None, phk=2.0, fit_func='ML',
                 wmat='normal'):
        if wmat == 'normal':
            W = linalg_utils.cov(Z, bias_corrected=True)
        elif wmat == 'wishart':
            W = linalg_utils.cov(Z, bias_corrected=False)
        elif wmat == 'adf':
            V = linalg_utils.adf_mat(Z)
            W = linalg_utils.cov(Z, bias_corrected=True)
            
        if fit_func == 'ML':
            self._obj_func = ObjFuncML(W=W, Winv=np.linalg.pinv(W), S=W)
        elif fit_func == 'QD':
            if wmat == 'adf':
                self._obj_func = ObjFuncQD(W=W, Winv=np.linalg.pinv(W), S=W, V=V)
            else:
                 self._obj_func = ObjFuncQD(W=W, Winv=np.linalg.pinv(W), S=W)
        elif fit_func == 'TR':
            self._obj_func = ObjFuncTR(W=W, Winv=np.linalg.pinv(W), S=W)
            
        if TH is not None:
            TH = linalg_utils._check_np(TH)
        if PH is not None:
            PH = linalg_utils._check_np(PH)
            
        Lmask = linalg_utils.omat(*LA.shape)
        Ltmp = LA.copy()
        dfd = collections.defaultdict(list) 
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
        Z, self.zcols, self.zix, self.z_is_pd = base_utils.check_type(Z)
        LA, self.lcols, self.lix, self.l_is_pd = base_utils.check_type(LA)
        BE, self.bcols, self.bix, self.b_is_pd = base_utils.check_type(BE)
        LA, idx1, BE, idx2, PH_i, idx3, TH_i, idx4 = self.init_params(Z, LA, BE, 
                                                                      TH, PH)
        if TH is None:
            TH = TH_i
        else:
            TH = TH
        PH = PH_i/phk
        p, k = LA.shape
        k1 = p * k #Number of Lambda params
        k2 = k * k # Number of Beta params
        k3 = int((k + 1) * k / 2) #Number of unique Phi params
        k4 = int((p + 1) * p / 2) #Number of unique theta Params
        
        #Cumulative sums
        k2 = k2 + k1
        k3 = k2 + k3
        k4 = k3 + k4 
        
        self.k1, self.k2, self.k3, self.k4 = k1, k2, k3, k4
        self.p, self.k = p, k
        self.n_obs = Z.shape[0]
        self.Z = Z
        self.S = linalg_utils.cov(Z) #True covariance
        self.LA = LA
        self.BE = BE
        self.IB = np.linalg.inv(linalg_utils.mat_rconj(BE))
        self.PH = PH
        self.TH = TH
        self.idx = self.mat_to_params(idx1, idx2, idx3, idx4) #Free parameter index
        self.params = self.mat_to_params(LA, BE, PH, TH)
        self.free = self.params[self.idx]
        self.Sigma = self.implied_cov(self.LA, self.BE, self.PH, self.TH)
        self.GLSW = linalg_utils.pre_post_elim(np.kron(np.linalg.inv(self.S),
                                                       np.linalg.inv(self.S)))
        self.Sinv = np.linalg.inv(self.S)
        self.Lp = linalg_utils.lmat(self.p)
        self.Np = linalg_utils.nmat(self.p)
        self.Ip = np.eye(self.p)
        self.Dk = linalg_utils.dmat(self.k)
        self.Dp = linalg_utils.dmat(self.p)
        self.Ip2 = np.eye(self.p**2)
        
        self.bounds = self.mat_to_params(linalg_utils.omat(*self.LA.shape), 
                                         linalg_utils.omat(*self.BE.shape),
                                         np.eye(self.PH.shape[0]),
                                         np.eye(self.TH.shape[0]))
        self.bounds = self.bounds[self.idx]
        self.bounds = [(None, None) if x==0 else (0, None) for x in self.bounds]
        
        
    def init_params(self, Z, L, B, TH=None, PH=None):
        BE_init = linalg_utils.omat(*B.shape)
        BE_idx = B.copy().astype(bool)
        LA_init = linalg_utils.omat(*L.shape)
        if TH is None:
            TH_init = linalg_utils.diag2(linalg_utils.cov(Z)) / 2
        else:
            TH_init = TH
        if PH is None:
            PH_init = np.eye(B.shape[0])*0.05
            PH_mask = np.eye(B.shape[0])
        else:
            PH_init = PH
            PH_mask = PH!=0
        dfd = collections.defaultdict(list) 
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
                LA_init[LA_idx[:, i], [i]] = linalg_utils.lstq(base_utils.center(exog), 
                       base_utils.center(endog)).flatten()
                Nu.append(linalg_utils.lstq_pred(base_utils.center(exog), 
                                                 base_utils.center(endog)))
            
        Nu = np.concatenate(Nu, axis=1) 
        
        for i in range(BE_idx.shape[0]):
            if np.sum(BE_idx[i])==0:
                continue
            else:
                exog = Nu[:, BE_idx[i]]
                endog = Nu[:, i]
            BE_init[i, BE_idx[i]] = linalg_utils.lstq(base_utils.center(exog), 
                   base_utils.center(endog))
        PH_init = linalg_utils.cov(Nu)*PH_mask
        PH_idx = PH_mask
        TH_idx = TH_init!=0
        return LA_init, LA_idx, BE_init, BE_idx, PH_init, PH_idx, TH_init, TH_idx

  
    def implied_cov(self, LA, BE, PH, TH):
        IB = np.linalg.inv(linalg_utils.mat_rconj(BE))
        Sigma = linalg_utils.mdot([LA, IB, PH, IB.T, LA.T]) + TH
        return Sigma
    
    def mat_to_params(self, LA, BE, PH, TH):
        params = np.block([linalg_utils.vec(LA), 
                           linalg_utils.vec(BE), 
                           linalg_utils.vech(PH), 
                           linalg_utils.vech(TH)])
        return params
  
    def get_mats(self, params=None):
        if params is None:
            params = self.params
        LA = linalg_utils.invec(params[:self.k1], self.p, self.k)
        BE = linalg_utils.invec(params[self.k1:self.k2], self.k, self.k)
        IB = np.linalg.inv(linalg_utils.mat_rconj(BE))
        PH = linalg_utils.invech(params[self.k2:self.k3])
        TH = linalg_utils.invech(params[self.k3:])
        return LA, BE, IB, PH, TH
    
    def obj_func(self, free):
        free = linalg_utils._check_1d(free)
        Sigma = self.get_sigma(free)
        return self._obj_func(Sigma)
    
    def gradient(self, free):
        free = linalg_utils._check_1d(free)
        Sigma = self.get_sigma(free)
        G = self.dsigma(free)
        g =  self._obj_func.gradient(Sigma, G)
        return g[:, 0][self.idx]
    
    def hessian(self, free):
        free = linalg_utils._check_1d(free)
        Sigma = self.get_sigma(free)
        G = self.dsigma(free)
        H =  self._obj_func.hessian(Sigma, G)
        H = H[self.idx][:, self.idx]
        return H
        
    
    def get_sigma(self, free):
        free = linalg_utils._check_1d(free)
        params = self.params.copy()
        if free.dtype==complex:
            params = params.astype(complex)
        params[self.idx] = free
        LA, BE, IB, PH, TH = self.get_mats(params)
        Sigma = self.implied_cov(LA, BE, PH, TH)
        return Sigma 
    
    def dsigma(self, free):
        params = self.params.copy()
        params[self.idx] = free
        LA, BE, IB, PH, TH = self.get_mats(params)
        A = np.dot(LA, IB)
        B = linalg_utils.mdot([A, PH, IB.T])
        DLambda = linalg_utils.mdot([self.Lp, self.Np, np.kron(B, self.Ip)])
        DBeta = linalg_utils.mdot([self.Lp, self.Np, np.kron(B, A)])
        DPhi = linalg_utils.mdot([self.Lp, np.kron(A, A), self.Dk])
        DPsi = linalg_utils.mdot([self.Lp, self.Ip2, self.Dp])        
        G = np.block([DLambda, DBeta, DPhi, DPsi])
        return G
    

    def einfo(self, free):
        params = self.params.copy()
        params[self.idx] = free
        Sigma = self.get_sigma(free)
        Sinv = np.linalg.inv(Sigma)
        D = linalg_utils.dmat(Sinv.shape[0])
        W = 2*linalg_utils.mdot([D.T, np.kron(Sinv, Sinv), D])
        G = self.dsigma(free)[:, self.idx]
        ncov = np.linalg.pinv(linalg_utils.mdot([G.T, W, G]))
        return ncov
    
    def robust_cov(self, free):
        mu = self.Z.mean(axis=0)
        Y = linalg_utils._check_np(self.Z)
        s = linalg_utils.vech(linalg_utils._check_np(self.S))
        ss = [linalg_utils.vech((Y[i] - mu)[:, None].dot((Y[i]-mu)[:, None].T)) 
              for i in range(Y.shape[0])]
        Gadf = np.sum([(si-s)[:, None].dot((si-s)[:, None].T) for si in ss],
                       axis=0)/Y.shape[0]
        
        Sigma = self.get_sigma(self.free)
        Sinv = np.linalg.inv(Sigma)
        D = linalg_utils.dmat(Sinv.shape[0])
        W = 2*linalg_utils.mdot([D.T, np.kron(Sinv, Sinv), D])
        G = self.dsigma(self.free)[:, self.idx]
        V = np.linalg.pinv(linalg_utils.mdot([G.T, W, G]))
        
        Vrob = V.dot(linalg_utils.mdot([G.T, W, Gadf, W, G])).dot(V)
        W /= 4.0
        U = W - W.dot(G).dot(np.linalg.inv(G.T.dot(W).dot(G)).dot(G.T).dot(W))
        scale = np.trace(U.dot(Gadf))
        return Vrob, scale

    
    def fit(self, method='ML', xtol=1e-20, gtol=1e-30, maxiter=3000, verbose=2):
        self.optimizer = sp.optimize.minimize(self.obj_func, self.free, 
                                  jac=self.gradient,
                                  hess=self.hessian, method='trust-constr',
                                  bounds=self.bounds,
                                  options={'xtol':xtol, 'gtol':gtol,
                                           'maxiter':maxiter,'verbose':verbose})    
        params = self.params.copy()
        params[self.idx] = self.optimizer.x           
        self.LA, self.BE, self.IB, self.PH, self.TH = self.get_mats(params)      
        self.free = self.optimizer.x      
        self.Sigma = self.get_sigma(self.free)
        
        self.SE_exp = 2*np.diag(self.einfo(self.free)/self.n_obs)**0.5
        self.SE_obs = np.diag(np.linalg.pinv(-self.hessian(self.free))/self.n_obs)**0.5
        Vrob, scale = self.robust_cov(self.free)
        self.SE_rob = np.sqrt(np.diag(Vrob)/75.0)
        self.res = pd.DataFrame([self.free, self.SE_exp, self.SE_obs, self.SE_rob], 
                                index=['Coefs','SE1', 'SE2', 'SEr'], 
                                columns=self.labels).T
        
        self.test_stat = (self.n_obs-1)*(self.obj_func(self.free)\
                         - np.linalg.slogdet(self.S)[1]-self.S.shape[0])
        self.df = len(linalg_utils.vech(self.S))-len(self.free)
        self.test_scale = scale / self.df
        self.t_robust = self.test_stat / self.test_scale
        self.test_pval = 1.0 - sp.stats.chi2.cdf(self.test_stat, self.df)
        self.robust_pval = sp.stats.chi2.sf(self.t_robust, self.df)
        self.res['t'] = self.res['Coefs'] / self.res['SE1']
        self.res['p'] = sp.stats.t.sf(abs(self.res['t']), self.n_obs)
        #self.res['adj p'] = fdr_bh(self.res['p'])
        self.SRMR = statfunc_utils.srmr(self.Sigma, self.S, self.df)
        self.GFI = statfunc_utils.gfi(self.Sigma, self.S)
        if self.df!=0:
            self.AGFI = statfunc_utils.agfi(self.Sigma, self.S, self.df)
            self.st_chi2 = (self.test_stat - self.df) / np.sqrt(2*self.df)
            self.RMSEA = np.sqrt(np.maximum(self.test_stat-self.df, 
                                     0)/(self.df*self.n_obs-1)) 
        else:
            self.AGFI = None
            self.st_chi2 = None
            self.RMSEA = None
        
        self.sumstats = [self.SRMR,
                         self.GFI,
                         self.AGFI,
                         self.st_chi2,
                         self.RMSEA,
                         self.test_stat,
                         self.test_pval,
                         self.t_robust,
                         self.robust_pval]
        self.sumstats = pd.DataFrame(self.sumstats)
        self.sumstats.index = ['SRMR', 'GFI', 'AGFI', 'chi2_st',
                               'RMSEA', 'chi2', 'chi2_pval', 'chi2_robust',
                               'chi2_robust_pval']
        self.sumstats.columns=['Goodness_of_fit']
        
        
'''        
        
      
data = pd.read_csv("/users/lukepinkel/Downloads/bollen.csv", index_col=0)
data = data[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]

L = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]])

B = np.array([[False, False, False],
              [True,  False, False],
              [True,  True, False]])
LA = pd.DataFrame(L, index=data.columns, columns=['ind60', 'dem60', 'dem65'])
BE = pd.DataFrame(B, index=LA.columns, columns=LA.columns)
S = data.cov()
Zg = ZR = data


Lambda=LA!=0
Beta=BE!=0 
Lambda, Beta = pd.DataFrame(Lambda), pd.DataFrame(Beta)
Lambda.columns = ['ind60', 'dem60', 'dem65']
Lambda.index = Zg.columns
Beta.columns = Lambda.columns
Beta.index = Lambda.columns

Theta = pd.DataFrame(np.eye(Lambda.shape[0]),
                     index=Lambda.index, columns=Lambda.index)
Theta.loc['y1', 'y5'] = 0.05
Theta.loc['y2', 'y4'] = 0.05
Theta.loc['y2', 'y6'] = 0.05
Theta.loc['y3', 'y7'] = 0.05
Theta.loc['y4', 'y8'] = 0.05
Theta.loc['y6', 'y8'] = 0.05

Theta.loc['y5', 'y1'] = 0.05
Theta.loc['y4', 'y2'] = 0.05
Theta.loc['y6', 'y2'] = 0.05
Theta.loc['y7', 'y3'] = 0.05
Theta.loc['y8', 'y4'] = 0.05
Theta.loc['y8', 'y6'] = 0.05


  
model1 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='ML', wmat='normal')
model1.fit()   

model2 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='ML', wmat='wishart')
model2.fit()


model3 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='QD', wmat='wishart')
model3.fit()   


model4 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='QD', wmat='normal')
model4.fit()   


model5 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='QD', wmat='adf')
model5.fit()   


model6 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='TR', wmat='wishart')
model6.fit()   


model7 = SEM(Zg, Lambda, Beta, Theta.values, fit_func='TR', wmat='normal')
model7.fit()   

res = pd.DataFrame(np.zeros((6, 7)), columns=['Function', 'Dist',
                   'SRMR', 'RMSEA', 'chi2', 'GFI', 'AGFI'])

ff1 = ['ML', 'ML', 'QD', 'QD', 'QD', 'TR', 'TR']
ff2 = ['norma', 'wishart', 'wishart', 'normal', 'adf', 'wishart', 'normal']
md = [model1, model2, model3, model4, model5, model6, model7]

for i in range(6):
    res.iloc[i, 0] = ff1[i]
    res.iloc[i, 1] = ff2[i]
    res.iloc[i, 2] = md[i].SRMR
    res.iloc[i, 3] = md[i].RMSEA
    res.iloc[i, 4] = md[i].test_stat
    res.iloc[i, 5] = md[i].GFI
    
model1.SRMR
model2.SRMR
model3.SRMR
model4.SRMR
model5.SRMR
model6.SRMR
model7.SRMR



'''


