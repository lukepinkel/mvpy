
import pandas as pd#analysis:ignore
import numpy as np #analysis:ignore
import scipy as sp #analysis:ignore
import scipy.stats #analysis:ignore
import collections#analysis:ignore
from ..utils import linalg_utils, base_utils, statfunc_utils #analysis:ignore

class MLSEM:
    """
    A little sloppy, but the hessian is finally correct for the full ML model
    """
 
    def __init__(self, Z, LA, BE, TH=None, PH=None, phk=2.0, fit_func='ML',
                 wmat='normal'):
       
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
            for x in LA_idx[LA_idx==True].T.stack().index.values:
                labels.append("%s ~ %s"%(x[1], x[0]))
        if (type(BE) is pd.DataFrame)|(type(BE) is pd.Series):
            for x in BE[BE==True].T.stack().index.values:
                labels.append("%s ~ %s"%(x[1], x[0]))   
        if PH is None:
            PH = np.eye(BE.shape[0])
        if (type(PH) is pd.DataFrame)|(type(PH) is pd.Series):
            for x in PH[PH!=0].stack().index.values:
                labels.append("var(%s ~ %s)"%(x[1], x[0]))
        else:
            tmp = pd.DataFrame(PH, index=LA.columns, columns=LA.columns)
            tix = np.triu(np.ones(tmp.shape)).astype('bool').reshape(tmp.size)
            tmp = tmp.stack()[tix]
            for x in tmp[tmp!=0].index.values:
                labels.append("var(%s, %s)"%(x[1], x[0])) 
                
                
        if (type(TH) is pd.DataFrame)|(type(TH) is pd.Series):
            for x in TH[TH!=0].stack().index.values:
                labels.append("resid(%s ~ %s)"%(x[1], x[0]))
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
        self.IB = np.linalg.pinv(linalg_utils.mat_rconj(BE))
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
        self.Kq = linalg_utils.kmat(self.k, self.k)
        self.Kp = linalg_utils.kmat(self.p, self.k)
        self.Dp = linalg_utils.dmat(self.p)
        self.T = np.zeros((self.p, self.p))
        self.Ip2 = np.eye(self.p**2)
        self._lndetS = np.linalg.slogdet(self.S)[1]
        self._llc = -self._lndetS-self.p
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
        IB = np.linalg.pinv(linalg_utils.mat_rconj(BE))
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
        IB = np.linalg.pinv(linalg_utils.mat_rconj(BE))
        PH = linalg_utils.invech(params[self.k2:self.k3])
        TH = linalg_utils.invech(params[self.k3:])
        return LA, BE, IB, PH, TH
    
    def loglike(self, free):
        free = linalg_utils._check_1d(free)
        Sigma = self.get_sigma(free)
        Sigma_inv = np.linalg.pinv(Sigma)
        LL = np.linalg.slogdet(Sigma)[1] + np.trace(self.S.dot(Sigma_inv))
        LL+= self._llc
        return LL
    
    def gradient(self, free):
        free = linalg_utils._check_1d(free)
        Sigma = self.get_sigma(free)
        Sigma_inv = np.linalg.pinv(Sigma)
        G = self.dsigma(free)
        W = 0.5 * self.Dp.T.dot(np.kron(Sigma_inv, Sigma_inv)).dot(self.Dp)
        d = linalg_utils.vechc(self.S - Sigma)
        g = -2.0 * G.T.dot(W).dot(d)
        return g[:, 0][self.idx]
    
    def _hessian_a(self, Sigma, Sinv, G):
        D = self.Dp
        H = 0.5 * linalg_utils.mdot([G.T, D.T, np.kron(Sinv, Sinv), D, G])
        return H 
    
    def _hessian_b(self, Sigma, Sinv, G):
        Sdiff = self.S - Sigma
        D = self.Dp
        H = G.T.dot(D.T.dot(np.kron(Sinv, Sinv.dot(Sdiff).dot(Sinv))).dot(D))
        H = H.dot(G)
        return H
    
    def _hessian_c(self, Sigma, Sinv, G, LA, IB, PH):
        Sdiff = self.S - Sigma
        d = linalg_utils.vech(Sdiff)
        Hpp = []
        U, A = IB.dot(PH).dot(IB.T), LA.dot(IB)
        Q = LA.dot(U)
        Kq, Kp, D, Dp, T = self.Kq, self.Kp, self.Dk, self.Dp, self.T
        Kp = linalg_utils.kmat(self.k, self.p)
        k1, k2, k3, p = self.k1, self.k2, self.k3, self.p
        for i in range(p):
            for j in range(i, p):
                Hij = np.zeros((len(self.params), len(self.params)))
                T[i, j] = T[j, i] = 1.
                TA = T.dot(A)
                AtTQ = A.T.dot(T).dot(Q)
                AtTA = A.T.dot(TA)
                
                H11 = np.kron(U, T)
                H22 = np.kron(AtTQ.T, IB.T).dot(Kq)+Kq.dot(np.kron(AtTQ, IB))\
                      +np.kron(U, AtTA)
                H12 = (np.kron(U, TA)) + Kp.dot(np.kron(T.dot(Q), IB))
                H13 =  np.kron(IB, TA).dot(D) 
                H23 = -D.T.dot(np.kron(IB.T, AtTA))
                Hij[:k1, :k1] = H11
                Hij[k1:k2, k1:k2] = H22
                Hij[:k1, k1:k2] = H12
                Hij[:k1, k2:k3] = H13
                Hij[k2:k3, k1:k2] = H23
                Hij[k1:k2, :k1] = H12.T
                Hij[k2:k3, :k1] = H13.T
                Hij[k1:k2, k2:k3] = H23.T  
                T[i, j] = T[j, i] = 0.0
                Hij = Hij[self.idx][:, self.idx]
                Hpp.append(Hij[:, :, None])
        W = np.linalg.multi_dot([Dp.T, np.kron(Sinv, Sinv), Dp])
        dW = np.dot(d, W)
        Hp = np.concatenate(Hpp, axis=2) 
        H = np.einsum('k,ijk ->ij', dW, Hp)
        return H
    
    def hessian(self, free):
        free = linalg_utils._check_1d(free)
        params = self.params.copy()
        if free.dtype==complex:
            params = params.astype(complex)
        params[self.idx] = free
        LA, BE, IB, PH, TH = self.get_mats(params)
        Sigma = self.get_sigma(free)
        Sinv = np.linalg.pinv(Sigma)
        G = self.dsigma(free)
        H1 = self._hessian_a(Sigma, Sinv, G)[self.idx][:, self.idx]
        H2 = self._hessian_b(Sigma, Sinv, G)[self.idx][:, self.idx]
        H3 = self._hessian_c(Sigma, Sinv, G, LA, IB, PH)
        H = H1 + H2 - H3/2
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
        free = linalg_utils._check_1d(free)
        params = self.params.copy()
        if free.dtype==complex:
            params = params.astype(complex)
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
        U = W - W.dot(G).dot(np.linalg.pinv(G.T.dot(W).dot(G)).dot(G.T).dot(W))
        scale = np.trace(U.dot(Gadf))
        return Vrob, scale

    
    def fit(self, method='ML', xtol=1e-20, gtol=1e-30, maxiter=3000, verbose=2,
            constraints=(), use_hess=False):
        if use_hess:
            
            hess = self.hessian
        else:
            hess = None
        self.optimizer = sp.optimize.minimize(self.loglike, self.free, 
                                  jac=self.gradient,
                                  constraints=constraints,
                                  hess=hess, method='trust-constr',
                                  bounds=self.bounds,
                                  options={'xtol':xtol, 'gtol':gtol,
                                           'maxiter':maxiter,'verbose':verbose})    
        params = self.params.copy()
        params[self.idx] = self.optimizer.x           
        self.LA, self.BE, self.IB, self.PH, self.TH = self.get_mats(params)      
        self.free = self.optimizer.x      
        self.Sigma = self.get_sigma(self.free)
        
        self.SE_exp = 2*np.diag(self.einfo(self.free)/self.n_obs)**0.5
        self.SE_obs = np.diag(np.linalg.pinv(self.hessian(self.free))/self.n_obs)**0.5
        Vrob, scale = self.robust_cov(self.free)
        self.SE_rob = np.sqrt(np.diag(Vrob)/75.0)
        self.res = pd.DataFrame([self.free, self.SE_exp, self.SE_obs, self.SE_rob], 
                                index=['Coefs','SE1', 'SE2', 'SEr'], 
                                columns=self.labels).T

        self.test_stat = (self.n_obs - 1) * (self.loglike(self.free))
        self.df = len(linalg_utils.vech(self.S))-len(self.free)
        
        self.test_scale = scale / self.df
        self.t_robust = self.test_stat / self.test_scale
        self.test_pval = 1.0 - sp.stats.chi2.cdf(self.test_stat, self.df)
        self.robust_pval = sp.stats.chi2.sf(self.t_robust, self.df)
        
        self.res['t'] = self.res['Coefs'] / self.res['SE1']
        self.res['p'] = sp.stats.t.sf(abs(self.res['t']), self.n_obs)
        self._goodness_of_fit()

    def _goodness_of_fit(self):
        SRMR = statfunc_utils.srmr(self.Sigma, self.S, self.df)
        GFI = statfunc_utils.gfi(self.Sigma, self.S)
        LL = (self.loglike(self.free) - self._llc) * (self.n_obs)/2 \
             + self.n_obs * self.p * np.log(2*np.pi) / 2.0
        test_stat = self.test_stat
        AIC = 2*len(self.free)+2*LL
        BIC = len(self.free)*np.log(self.n_obs)+2*LL
        Sb = np.diag(np.diag(self.S))
        tbase = (np.linalg.slogdet(Sb)[1] \
                + np.trace(self.S.dot(np.linalg.inv(Sb)))) * (self.n_obs - 1)
        NFI1 = (tbase - test_stat) / tbase
        dfm = len(self.free) 
        dfb = self.S.shape[0] 
        NFI2 = (tbase - test_stat) / (tbase - dfm)
        RhoFI1 = (tbase/dfb - test_stat/dfm) / (tbase/dfb)
        RhoFI2 = (tbase/dfb - test_stat/dfm)/(tbase/dfb-1)
        if self.df!=0:
            AGFI = statfunc_utils.agfi(self.Sigma, self.S, self.df)
            st_chi2 = (test_stat - self.df) / np.sqrt(2*self.df)
            RMSEA = np.sqrt(np.maximum(self.test_stat-self.df, 
                                     0)/(self.df*self.n_obs-1)) 
        else:
            AGFI, st_chi2, RMSEA = None, None, None
        rsquared = 1 - np.diag(self.TH) / np.diag(self.S)
        self.LL = LL
        self.rsquared = pd.DataFrame(rsquared, index=self.zcols)
        self.r2total = 1 - np.linalg.det(self.TH) / np.linalg.det(self.S)
        self.sumstats = pd.DataFrame([[AGFI, '-'],
                                      [AIC, '-'],
                                      [BIC, '-'],
                                      [GFI, '-'],
                                      [NFI1, '-'],
                                      [NFI2, '-'],
                                      [RhoFI1, '-'],
                                      [RhoFI2, '-'],
                                      [RMSEA, '-'],
                                      [SRMR, '-'],
                                      [test_stat, self.test_pval],
                                      [self.t_robust, self.robust_pval],
                                      [st_chi2, '-']
                                      ])
        self.sumstats.index = ['AGFI', 'AIC', 'BIC', 'GFI', 'NFI1', 
                               'NFI2', 'RhoFI1', 'RhoFI2', 'RMSEA', 'SRMR',
                               'chi2', 'chi2_robust', 'chi2_standard']
        self.sumstats.columns=['Goodness_of_fit', 'P value']
        
        
        
        