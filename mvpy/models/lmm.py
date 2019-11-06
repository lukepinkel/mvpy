#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:03:34 2019

@author: lukepinkel
"""
import patsy
import collections
import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats
import scipy.sparse as sps

from ..utils import linalg_utils, data_utils


class LMM(object):

    def __init__(self, fixed_effects, random_effects, yvar, data,
                 error_structure=None, acov=None):
        '''
        Linear Mixed Model

        Parameters
        ----------
        fixed_effects: str
            String formula containing the column names corresponding to the
            fixed effect terms. This is to be evaluated by patsy

        random_effects: dict
            Dictionary whose keys correspond to factors, and whose values are
            formulas specifying the random effect terms.  The keys are evaluated
            as categorical and dummy encoded.

        y: str or list
            If y is a string, then a univariate linear mixed model is fit, but if
            y is a list of m strings, then an m-variate linear mixed model is
            fit. It is assumed that all fixed and random effects are identical
            for all dependent variables.

        data: DataFrame
            Pandas DataFrame containing n_obs by n_features including the
            relavent terms to the model

        error_structure: str, default None
            Error structure defaults to iid, but a factor level may be provided
            via a string referencing a column name, which will then be used to
            constructthe error covariance.  Implemented for multivariate linear
            models, where it is repeated across the multiple dependent variables,
            and has the structure Cov(Error) = V_{m\\times m}\\otimes I_{n}
        acov: dict, default None
            Similar to random_effects, dictionary with keys indicating factors
            except the values need to be matrices that specify the covariance
            among observational units (row covariance)

        '''

        n_obs = data.shape[0]
        X = patsy.dmatrix(fixed_effects, data=data, return_type='dataframe')
        fixed_effects = X.columns
        Z = []
        re_struct = collections.OrderedDict()
        
        #Determine if model is multivariate
        if type(yvar) is list: 
            n_vars = len(yvar)
            yvnames = yvar
        else:
            n_vars = 1
            yvnames = [yvar]
         
        res_names = [] #might be better of renamed re_names; may be typo
        for key in random_effects.keys():
            #dummy encode the groupings and get random effect variate
            Ji = data_utils.dummy_encode(data[key], complete=True)
            Zij = patsy.dmatrix(random_effects[key], data=data,
                                return_type='dataframe')
            #stratify re variable by dummy columns
            Zi = linalg_utils.khatri_rao(Ji.T, Zij.T).T 
            Z.append(Zi)
            k = Zij.shape[1]*n_vars
            # RE dependence structure
            if (acov is not None):
                if acov[key] is not None: #dependence structure for each RE
                    acov_i = acov[key]
                else:                     #single dependence for all REs
                    acov_i = np.eye(Ji.shape[1])
            else:                         #IID
                acov_i = np.eye(Ji.shape[1])
            re_struct[key] = {'n_units': Ji.shape[1],
                              'n_level_effects': Zij.shape[1],
                              'cov_re_dims': k,
                              'n_params': ((k + 1.0) * k) / 2.0,
                              'vcov': np.eye(k),
                              'params': linalg_utils.vech(np.eye(k)),
                              'acov': acov_i}
            if len(yvnames)>1:
                names = [x+": "+y for x in yvnames for y in
                         Zij.columns.tolist()]
                names = np.array(names)
            else:  
                names = np.array(Zij.columns.tolist())
            names_a = names[np.triu_indices(k)[0]]
            names_b = names[np.triu_indices(k)[1]]
            for r in range(len(names_a)):
                res_names.append(key+'|'+names_a[r]+' x '+names_b[r])
        

        Z = np.concatenate(Z, axis=1)

        error_struct = collections.OrderedDict()
        error_struct['vcov'] = np.eye(n_vars)
        error_struct['acov'] = np.eye(n_obs)
        error_struct['params'] = linalg_utils.vech(np.eye(n_vars))
        if len(yvnames)>1&(type(yvnames) is list):
            tmp = []
            for i, x in enumerate(yvnames):
                for j, y in enumerate(yvnames):
                    if i <= j:
                        tmp.append(x+": "+y+" error_var")
            res_names += tmp
        else:  
            res_names += ['error_var']
            
        if type(yvar) is str:
            y = data[[yvar]]
        
        #Vectorize equations - Add flexibility for dependent variable specific
        #design matrices
        elif type(yvar) is list:
            y = linalg_utils.vecc(data[yvar].values)
            Z = np.kron(np.eye(n_vars), Z)
            X = np.vstack([X for i in range(n_vars)])

        var_params = np.concatenate([re_struct[key]["params"]
                                     for key in re_struct.keys()])
        err_params = error_struct['params']
        partitions = [0]+[re_struct[key]['n_params']
                          for key in re_struct.keys()]
        partitions += [len(error_struct['params'])]
        theta = np.concatenate([var_params, err_params])
        partitions2 = [0]+[re_struct[key]['n_units']
                           * re_struct[key]['cov_re_dims']
                           for key in re_struct.keys()]
        partitions2 = np.cumsum(partitions2)
        var_struct = collections.OrderedDict()
        for key in re_struct.keys():
            var_struct[key] = [re_struct[key]['vcov'].shape,
                               re_struct[key]['acov']]
        var_struct['error'] = [error_struct['vcov'].shape,
                               error_struct['acov']]
        #Get Z and Z otimes Z for each RE
        Zs = collections.OrderedDict()
        ZoZ = collections.OrderedDict()
        for i in range(len(re_struct)):
            key = list(re_struct)[i]
            Zs[key] = sps.csc_matrix(Z[:, partitions2[i]:partitions2[i+1]])
            ZoZ[key] = sps.csc_matrix(sps.kron(Zs[key], Zs[key]))

        deriv_mats = collections.OrderedDict()
        for key in var_struct.keys():
            Sv_shape, Av = var_struct[key]
            Av_shape = Av.shape
            Kv = linalg_utils.kronvec_mat(Sv_shape, Av_shape, sparse=True)
            Ip = sps.csc_matrix(sps.eye(np.product(Sv_shape)))
            vecAv = sps.csc_matrix(linalg_utils.vecc(Av))

            D = sps.csc_matrix(Kv.dot(sps.kron(Ip, vecAv)))
            if key != 'error':
                D = sps.csc_matrix(ZoZ[key].dot(D))
            tmp = sps.csc_matrix(linalg_utils.dmat(int(np.sqrt(D.shape[1]))))
            deriv_mats[key] = D.dot(tmp)

        self.var_struct = var_struct
        self.deriv_mats = deriv_mats
        self.bounds = [(0, None) if x == 1 else (None, None) for x in theta]
        self.theta = theta
        self.partitions = np.cumsum(partitions)
        J = sps.hstack([deriv_mats[key] for key in deriv_mats])
        self.jac_mats = [J[:, i].reshape(Z.shape[0], Z.shape[0], order='F')
                         for i in range(J.shape[1])]

        self.X = linalg_utils._check_np(X)
        self.Z = linalg_utils._check_np(Z)
        self.y = linalg_utils._check_np(y)
        self.error_struct = error_struct
        self.re_struct = re_struct
        self.ZoZ = ZoZ
        self.res_names = res_names + fixed_effects.tolist()
        self.n_vars = n_vars
        self.XZY = np.block([X, Z, y])
        self.XZ = np.block([X, Z])
        self.A = np.block([[X, Z], [np.zeros((Z.shape[1], X.shape[1])),
                           np.eye(Z.shape[1])]])

    def params2mats(self, theta=None):
        '''
        Create variance matrices from parameter vector
        Parameters
        ------------
        theta: array
            Vector containing relavent model terms
        '''
        if theta is None:
            theta = self.theta
        partitions = self.partitions
        error_struct = self.error_struct
        re_struct = self.re_struct

        Glist, Ginvlist, SigA = [], [], []
        for i, key in enumerate(re_struct.keys()):
            a, b = int(partitions[i]), int(partitions[i+1])
            Vi = linalg_utils.invech(theta[a:b])
            Ai = re_struct[key]['acov']
            Glist.append(np.kron(Vi, Ai))
            Ginvlist.append(np.kron(np.linalg.pinv(Vi), Ai))
            SigA.append(Vi)
        p1, p2 = int(partitions[-2]), int(partitions[-1])
        Verr = linalg_utils.invech(theta[p1:p2])
        R = np.kron(Verr, error_struct['acov'])
        Rinv = np.kron(np.linalg.inv(Verr), error_struct['acov'])
        G, Ginv = sp.linalg.block_diag(*Glist), sp.linalg.block_diag(*Ginvlist)

        SigE = Verr.copy()
        return G, Ginv, SigA, R, Rinv, SigE

    def mmec(self, Rinv, Ginv):
        '''
        Mixed Model Equation Coefficient(MMEC) matrix construction
        Parameters
        ------------
        Rinv: array
          Inverse error covariance
        Ginv:
          Inverse random effect covariance
        '''
        F = self.XZ
        C = F.T.dot(Rinv).dot(F)
        k = Ginv.shape[0]
        C[-k:, -k:] += Ginv
        return C

    def mme_aug(self, Rinv, Ginv, C=None):
        '''
        Augmented Mixed Model Equation Coefficient matrix construction
        Parameters
        ------------
        Rinv: array
          Inverse error covariance
        Ginv: array
          Inverse random effect covariance
        C: array
          MMEC coefficient matrix

        '''
        if C is None:
            C = self.mmec(Rinv, Ginv)
        XZ, y = self.XZ, self.y
        t = y.T.dot(Rinv)
        b = t.dot(XZ)
        yRy = linalg_utils._check_np(t).dot(y)
        M = np.block([[C, b.T], [b, yRy]])
        return M

    def loglike(self, theta):
        '''
        Minus two times the restricted log likelihood
        Parameters
        ---------
        theta: array
            vector of parameters
        '''
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        re_struct, error_struct = self.re_struct, self.error_struct
        C = self.mmec(Rinv, Ginv)
        M = self.mme_aug(Rinv, Ginv, C=C)
        L = linalg_utils.chol(M)
        logdetC = 2*np.sum(np.log(np.diag(L)[:-1]))
        yPy = L[-1, -1]**2
        logdetG = 0.0
        for key, Vi in list(zip(re_struct.keys(), SigA)):
            logdetG += re_struct[key]['n_units']*np.linalg.slogdet(Vi)[1]
        logdetR = error_struct['acov'].shape[0]*np.linalg.slogdet(SigE)[1]
        LL = logdetR+logdetC + logdetG + yPy
        return LL

    def fit(self, optimizer_kwargs={}, maxiter=100, verbose=2, hess_opt=False):
        if hess_opt is False:
            res = sp.optimize.minimize(self.loglike, self.theta,
                                       bounds=self.bounds,
                                       options={'verbose': verbose,
                                                'maxiter': maxiter},
                                       method='trust-constr',
                                       jac=self.gradient,
                                       **optimizer_kwargs)
        else:
            res = sp.optimize.minimize(self.loglike, self.theta,
                                       bounds=self.bounds,
                                       options={'verbose': verbose,
                                                'maxiter': maxiter},
                                       method='trust-constr',
                                       jac=self.gradient,
                                       hess=self.hessian,
                                       **optimizer_kwargs)

        self.params = res.x
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(res.x)
        self.G, self.Ginv, self.R, self.Rinv = G, Ginv, R, Rinv
        self.SigA, self.SigE = SigA, SigE
        W = linalg_utils.woodbury_inversion(self.Z, C=G, A=R)
        X = self.X
        XtW = X.T.dot(W)
        self.optimizer = res
        self.hessian_est = self.hessian(self.params)
        self.hessian_inv = np.linalg.pinv(self.hessian_est)
        self.SE_theta = np.sqrt(np.diag(self.hessian_inv))
        self.grd = self.gradient(self.params)
        self.gnorm = np.linalg.norm(self.grd) / len(self.params)
        self.b = linalg_utils.einv(XtW.dot(X)).dot(XtW.dot(self.y))
        self.SE_b = np.sqrt(np.diag(linalg_utils.einv(XtW.dot(X))))
        self.r = self.y - self.X.dot(self.b)
        self.u = G.dot(self.Z.T.dot(W).dot(self.r))
        res = pd.DataFrame(np.concatenate([self.params[:, None], self.b]),
                           columns=['Parameter Estimate'])
        res['Standard Error'] = np.concatenate([self.SE_theta, self.SE_b])
        res['t value'] = res['Parameter Estimate'] / res['Standard Error']
        res['p value'] = sp.stats.t.sf(np.abs(res['t value']),
                                       X.shape[0]-len(self.params)) * 2.0
        res.index = self.res_names
        self.res = res
        n_obs, k_params = self.X.shape[0], len(self.params)
        
        self.ll = self.loglike(self.params)
        self.aic = self.ll + (2 * k_params)
        self.aicc = self.ll + 2*k_params*n_obs / (n_obs - k_params - 1)
        self.bic = self.ll + k_params*np.log(n_obs)
        self.caic = self.ll + k_params * np.log(n_obs+1)
        self.r2_fe = 1 - np.var(self.y - self.X.dot(self.b)) / np.var(self.y)
        self.r2_re = 1 - np.var(self.y - self.Z.dot(self.u)) / np.var(self.y)
        self.r2 = 1 - np.var(self.y - self.predict()) / np.var(self.y)
        self.sumstats = np.array([self.aic, self.aicc, self.bic, self.caic,
                                  self.r2_fe, self.r2_re, self.r2])
        self.sumstats = pd.DataFrame(self.sumstats, index=['AIC', 'AICC', 'BIC',
                                                           'CAIC', 
                                                           'FixedEffectsR2',
                                                           'RandomEffectsR2', 
                                                           'R2'])
        
    def predict(self, X=None, Z=None):
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        return X.dot(self.b)+Z.dot(self.u)

    def gradient(self, theta):
        '''
        The gradient of minus two times the restricted log likelihood.  This is
        equal to

        \\partial\\mathcal{L}=vec(Py)'\\partial V-(vec(Py)\\otimes
                              vec(Py))'\\partial V

        Parameters
        ----------
        theta: array
          Vector of parameters

        Returns
        --------
        g: array
          gradient vector of one dimensions (for compatibility with minimize)

        '''
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        deriv_mats = self.deriv_mats
        X, Z, y = self.X, self.Z, self.y
        W = linalg_utils.woodbury_inversion(Z, Cinv=Ginv, Ainv=Rinv) 
        XtW = X.T.dot(W)
        XtWX_inv = linalg_utils.einv(XtW.dot(X))
        P = W - XtW.T.dot(XtWX_inv).dot(XtW)
        dP = P.reshape(np.product(P.shape), 1, order='F')
        Py = P.dot(y)
        PyPy = np.kron(Py, Py)
        # PyPy = vec(_check_2d(Py).dot(_check_2d(Py).T))[:, None] effecient
        # only at large heterogenous n
        g = []
        for key in deriv_mats.keys():
            JF_Omega = deriv_mats[key]
            g_i = JF_Omega.T.dot(dP) - JF_Omega.T.dot(PyPy)
            g.append(g_i)
        g = np.concatenate(g)
        return linalg_utils._check_1d(g)

    def hessian(self, theta):
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        jac_mats = self.jac_mats
        X, Z, y = self.X, self.Z, self.y
        W = linalg_utils.woodbury_inversion(Z, Cinv=Ginv, Ainv=Rinv)
        XtW = X.T.dot(W)
        XtWX_inv = linalg_utils.einv(XtW.dot(X))
        P = W - XtW.T.dot(XtWX_inv).dot(XtW)
        # P = W - np.linalg.multi_dot([XtW.T, XtWX_inv, XtW])
        Py = P.dot(y)
        H = []
        for i, Ji in enumerate(jac_mats):
            for j, Jj in enumerate(jac_mats):
                if j >= i:
                    PJi = sps.coo_matrix.dot(Ji.T, P).T
                    PJj = sps.coo_matrix.dot(Jj.T, P).T
                    yPJi = sps.coo_matrix.dot(Ji.T, Py).T
                    JjPy = Jj.dot(Py)
                    Hij = -(PJi.dot(PJj)).diagonal().sum()\
                        + (2 * (yPJi.dot(P)).dot(JjPy))[0]
                    H.append(Hij[0])
        H = linalg_utils.invech(np.array(H))
        return H
    

class GLMM(object):

    def __init__(self, fixed_effects, random_effects, yvar, data, W, 
                 error_structure=None, acov=None):
        '''
        Linear Mixed Model

        Parameters
        ----------
        fixed_effects: str
            String formula containing the column names corresponding to the
            fixed effect terms. This is to be evaluated by patsy

        random_effects: dict
            Dictionary whose keys correspond to factors, and whose values are
            formulas specifying the random effect terms.  The keys are evaluated
            as categorical and dummy encoded.

        y: str or list
            If y is a string, then a univariate linear mixed model is fit, but if
            y is a list of m strings, then an m-variate linear mixed model is
            fit. It is assumed that all fixed and random effects are identical
            for all dependent variables.

        data: DataFrame
            Pandas DataFrame containing n_obs by n_features including the
            relavent terms to the model

        error_structure: str, default None
            Error structure defaults to iid, but a factor level may be provided
            via a string referencing a column name, which will then be used to
            constructthe error covariance.  Implemented for multivariate linear
            models, where it is repeated across the multiple dependent variables,
            and has the structure Cov(Error) = V_{m\\times m}\\otimes I_{n}
        acov: dict, default None
            Similar to random_effects, dictionary with keys indicating factors
            except the values need to be matrices that specify the covariance
            among observational units (row covariance)

        '''
        self.W = W
        self.Winv = np.linalg.inv(W)
        n_obs = data.shape[0]
        X = patsy.dmatrix(fixed_effects, data=data, return_type='dataframe')
        fixed_effects = X.columns
        Z = []
        re_struct = collections.OrderedDict()
        
        #Determine if model is multivariate
        if type(yvar) is list: 
            n_vars = len(yvar)
            yvnames = yvar
        else:
            n_vars = 1
            yvnames = [yvar]
         
        res_names = [] #might be better of renamed re_names; may be typo
        for key in random_effects.keys():
            #dummy encode the groupings and get random effect variate
            Ji = data_utils.dummy_encode(data[key], complete=True)
            Zij = patsy.dmatrix(random_effects[key], data=data,
                                return_type='dataframe')
            #stratify re variable by dummy columns
            Zi = linalg_utils.khatri_rao(Ji.T, Zij.T).T 
            Z.append(Zi)
            k = Zij.shape[1]*n_vars
            # RE dependence structure
            if (acov is not None):
                if acov[key] is not None: #dependence structure for each RE
                    acov_i = acov[key]
                else:                     #single dependence for all REs
                    acov_i = np.eye(Ji.shape[1])
            else:                         #IID
                acov_i = np.eye(Ji.shape[1])
            re_struct[key] = {'n_units': Ji.shape[1],
                              'n_level_effects': Zij.shape[1],
                              'cov_re_dims': k,
                              'n_params': ((k + 1.0) * k) / 2.0,
                              'vcov': np.eye(k),
                              'params': linalg_utils.vech(np.eye(k)),
                              'acov': acov_i}
            if len(yvnames)>1:
                names = [x+": "+y for x in yvnames for y in
                         Zij.columns.tolist()]
                names = np.array(names)
            else:  
                names = np.array(Zij.columns.tolist())
            names_a = names[np.triu_indices(k)[0]]
            names_b = names[np.triu_indices(k)[1]]
            for r in range(len(names_a)):
                res_names.append(key+'|'+names_a[r]+' x '+names_b[r])
        

        Z = np.concatenate(Z, axis=1)

        error_struct = collections.OrderedDict()
        error_struct['vcov'] = np.eye(n_vars)
        error_struct['acov'] = np.eye(n_obs)
        error_struct['params'] = linalg_utils.vech(np.eye(n_vars))
        if len(yvnames)>1&(type(yvnames) is list):
            tmp = []
            for i, x in enumerate(yvnames):
                for j, y in enumerate(yvnames):
                    if i <= j:
                        tmp.append(x+": "+y+" error_var")
            res_names += tmp
        else:  
            res_names += ['error_var']
            
        if type(yvar) is str:
            y = data[[yvar]]
        
        #Vectorize equations - Add flexibility for dependent variable specific
        #design matrices
        elif type(yvar) is list:
            y = linalg_utils.vecc(data[yvar].values)
            Z = np.kron(np.eye(n_vars), Z)
            X = np.vstack([X for i in range(n_vars)])

        var_params = np.concatenate([re_struct[key]["params"]
                                     for key in re_struct.keys()])
        err_params = error_struct['params']
        partitions = [0]+[re_struct[key]['n_params']
                          for key in re_struct.keys()]
        partitions += [len(error_struct['params'])]
        theta = np.concatenate([var_params, err_params])
        partitions2 = [0]+[re_struct[key]['n_units']
                           * re_struct[key]['cov_re_dims']
                           for key in re_struct.keys()]
        partitions2 = np.cumsum(partitions2)
        var_struct = collections.OrderedDict()
        for key in re_struct.keys():
            var_struct[key] = [re_struct[key]['vcov'].shape,
                               re_struct[key]['acov']]
        var_struct['error'] = [error_struct['vcov'].shape,
                               error_struct['acov']]
        #Get Z and Z otimes Z for each RE
        Zs = collections.OrderedDict()
        ZoZ = collections.OrderedDict()
        for i in range(len(re_struct)):
            key = list(re_struct)[i]
            Zs[key] = sps.csc_matrix(Z[:, partitions2[i]:partitions2[i+1]])
            ZoZ[key] = sps.csc_matrix(sps.kron(Zs[key], Zs[key]))

        deriv_mats = collections.OrderedDict()
        for key in var_struct.keys():
            Sv_shape, Av = var_struct[key]
            Av_shape = Av.shape
            Kv = linalg_utils.kronvec_mat(Sv_shape, Av_shape, sparse=True)
            Ip = sps.csc_matrix(sps.eye(np.product(Sv_shape)))
            vecAv = sps.csc_matrix(linalg_utils.vecc(Av))

            D = sps.csc_matrix(Kv.dot(sps.kron(Ip, vecAv)))
            if key != 'error':
                D = sps.csc_matrix(ZoZ[key].dot(D))
            tmp = sps.csc_matrix(linalg_utils.dmat(int(np.sqrt(D.shape[1]))))
            deriv_mats[key] = D.dot(tmp)

        self.var_struct = var_struct
        self.deriv_mats = deriv_mats
        self.bounds = [(0, None) if x == 1 else (None, None) for x in theta]
        self.theta = theta
        self.partitions = np.cumsum(partitions)
        J = sps.hstack([deriv_mats[key] for key in deriv_mats])
        self.jac_mats = [J[:, i].reshape(Z.shape[0], Z.shape[0], order='F')
                         for i in range(J.shape[1])]

        self.X = linalg_utils._check_np(X)
        self.Z = linalg_utils._check_np(Z)
        self.y = linalg_utils._check_np(y)
        self.error_struct = error_struct
        self.re_struct = re_struct
        self.ZoZ = ZoZ
        self.res_names = res_names + fixed_effects.tolist()
        self.n_vars = n_vars
        self.XZY = np.block([X, Z, y])
        self.XZ = np.block([X, Z])
        self.A = np.block([[X, Z], [np.zeros((Z.shape[1], X.shape[1])),
                           np.eye(Z.shape[1])]])

    def params2mats(self, theta=None):
        '''
        Create variance matrices from parameter vector
        Parameters
        ------------
        theta: array
            Vector containing relavent model terms
        '''
        if theta is None:
            theta = self.theta
        partitions = self.partitions
        error_struct = self.error_struct
        re_struct = self.re_struct

        Glist, Ginvlist, SigA = [], [], []
        for i, key in enumerate(re_struct.keys()):
            a, b = int(partitions[i]), int(partitions[i+1])
            Vi = linalg_utils.invech(theta[a:b])
            Ai = re_struct[key]['acov']
            Glist.append(np.kron(Vi, Ai))
            Ginvlist.append(np.kron(np.linalg.pinv(Vi), Ai))
            SigA.append(Vi)
        p1, p2 = int(partitions[-2]), int(partitions[-1])
        Verr = linalg_utils.invech(theta[p1:p2])
        R = np.kron(Verr, error_struct['acov'])
        W, Winv = self.W, self.Winv
        R = W.dot(R).dot(W)
        Rinv = np.kron(np.linalg.inv(Verr), error_struct['acov'])
        Rinv = Winv.dot(Rinv).dot(Winv)
        G, Ginv = sp.linalg.block_diag(*Glist), sp.linalg.block_diag(*Ginvlist)

        SigE = Verr.copy()
        return G, Ginv, SigA, R, Rinv, SigE

    def mmec(self, Rinv, Ginv):
        '''
        Mixed Model Equation Coefficient(MMEC) matrix construction
        Parameters
        ------------
        Rinv: array
          Inverse error covariance
        Ginv:
          Inverse random effect covariance
        '''
        F = self.XZ
        C = F.T.dot(Rinv).dot(F)
        k = Ginv.shape[0]
        C[-k:, -k:] += Ginv
        return C

    def mme_aug(self, Rinv, Ginv, C=None):
        '''
        Augmented Mixed Model Equation Coefficient matrix construction
        Parameters
        ------------
        Rinv: array
          Inverse error covariance
        Ginv: array
          Inverse random effect covariance
        C: array
          MMEC coefficient matrix

        '''
        if C is None:
            C = self.mmec(Rinv, Ginv)
        XZ, y = self.XZ, self.y
        t = y.T.dot(Rinv)
        b = t.dot(XZ)
        yRy = linalg_utils._check_np(t).dot(y)
        M = np.block([[C, b.T], [b, yRy]])
        return M

    def loglike(self, theta):
        '''
        Minus two times the restricted log likelihood
        Parameters
        ---------
        theta: array
            vector of parameters
        '''
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        re_struct, error_struct = self.re_struct, self.error_struct
        C = self.mmec(Rinv, Ginv)
        M = self.mme_aug(Rinv, Ginv, C=C)
        L = linalg_utils.chol(M)
        logdetC = 2*np.sum(np.log(np.diag(L)[:-1]))
        yPy = L[-1, -1]**2
        logdetG = 0.0
        for key, Vi in list(zip(re_struct.keys(), SigA)):
            logdetG += re_struct[key]['n_units']*np.linalg.slogdet(Vi)[1]
        logdetR = error_struct['acov'].shape[0]*np.linalg.slogdet(SigE)[1]
        LL = logdetR+logdetC + logdetG + yPy
        return LL

    def fit(self, optimizer_kwargs={}, maxiter=100, verbose=2, hess_opt=False):
        if hess_opt is False:
            res = sp.optimize.minimize(self.loglike, self.theta,
                                       bounds=self.bounds,
                                       options={'verbose': verbose,
                                                'maxiter': maxiter},
                                       method='trust-constr',
                                       jac=self.gradient,
                                       **optimizer_kwargs)
        else:
            res = sp.optimize.minimize(self.loglike, self.theta,
                                       bounds=self.bounds,
                                       options={'verbose': verbose,
                                                'maxiter': maxiter},
                                       method='trust-constr',
                                       jac=self.gradient,
                                       hess=self.hessian,
                                       **optimizer_kwargs)

        self.params = res.x
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(res.x)
        self.G, self.Ginv, self.R, self.Rinv = G, Ginv, R, Rinv
        self.SigA, self.SigE = SigA, SigE
        W = linalg_utils.woodbury_inversion(self.Z, C=G, A=R)
        X = self.X
        XtW = X.T.dot(W)
        self.optimizer = res
        self.hessian_est = self.hessian(self.params)
        self.hessian_inv = np.linalg.pinv(self.hessian_est)
        self.SE_theta = np.sqrt(np.diag(self.hessian_inv))
        self.grd = self.gradient(self.params)
        self.gnorm = np.linalg.norm(self.grd) / len(self.params)
        self.b = linalg_utils.einv(XtW.dot(X)).dot(XtW.dot(self.y))
        self.SE_b = np.sqrt(np.diag(linalg_utils.einv(XtW.dot(X))))
        self.r = self.y - self.X.dot(self.b)
        self.u = G.dot(self.Z.T.dot(W).dot(self.r))
        res = pd.DataFrame(np.concatenate([self.params[:, None], self.b]),
                           columns=['Parameter Estimate'])
        res['Standard Error'] = np.concatenate([self.SE_theta, self.SE_b])
        res['t value'] = res['Parameter Estimate'] / res['Standard Error']
        res['p value'] = sp.stats.t.sf(np.abs(res['t value']),
                                       X.shape[0]-len(self.params)) * 2.0
        res.index = self.res_names
        self.res = res
        n_obs, k_params = self.X.shape[0], len(self.params)
        
        self.ll = self.loglike(self.params)
        self.aic = self.ll + (2 * k_params)
        self.aicc = self.ll + 2*k_params*n_obs / (n_obs - k_params - 1)
        self.bic = self.ll + k_params*np.log(n_obs)
        self.caic = self.ll + k_params * np.log(n_obs+1)
        self.r2_fe = 1 - np.var(self.y - self.X.dot(self.b)) / np.var(self.y)
        self.r2_re = 1 - np.var(self.y - self.Z.dot(self.u)) / np.var(self.y)
        self.r2 = 1 - np.var(self.y - self.predict()) / np.var(self.y)
        self.sumstats = np.array([self.aic, self.aicc, self.bic, self.caic,
                                  self.r2_fe, self.r2_re, self.r2])
        self.sumstats = pd.DataFrame(self.sumstats, index=['AIC', 'AICC', 'BIC',
                                                           'CAIC', 
                                                           'FixedEffectsR2',
                                                           'RandomEffectsR2', 
                                                           'R2'])
        
    def predict(self, X=None, Z=None):
        if X is None:
            X = self.X
        if Z is None:
            Z = self.Z
        return X.dot(self.b)+Z.dot(self.u)

    def gradient(self, theta):
        '''
        The gradient of minus two times the restricted log likelihood.  This is
        equal to

        \\partial\\mathcal{L}=vec(Py)'\\partial V-(vec(Py)\\otimes
                              vec(Py))'\\partial V

        Parameters
        ----------
        theta: array
          Vector of parameters

        Returns
        --------
        g: array
          gradient vector of one dimensions (for compatibility with minimize)

        '''
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        deriv_mats = self.deriv_mats
        X, Z, y = self.X, self.Z, self.y
        W = linalg_utils.woodbury_inversion(Z, Cinv=Ginv, Ainv=Rinv) 
        XtW = X.T.dot(W)
        XtWX_inv = linalg_utils.einv(XtW.dot(X))
        P = W - XtW.T.dot(XtWX_inv).dot(XtW)
        dP = P.reshape(np.product(P.shape), 1, order='F')
        Py = P.dot(y)
        PyPy = np.kron(Py, Py)
        # PyPy = vec(_check_2d(Py).dot(_check_2d(Py).T))[:, None] effecient
        # only at large heterogenous n
        g = []
        for key in deriv_mats.keys():
            JF_Omega = deriv_mats[key]
            g_i = JF_Omega.T.dot(dP) - JF_Omega.T.dot(PyPy)
            g.append(g_i)
        g = np.concatenate(g)
        return linalg_utils._check_1d(g)

    def hessian(self, theta):
        theta = linalg_utils._check_1d(theta)
        G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
        jac_mats = self.jac_mats
        X, Z, y = self.X, self.Z, self.y
        W = linalg_utils.woodbury_inversion(Z, Cinv=Ginv, Ainv=Rinv)
        XtW = X.T.dot(W)
        XtWX_inv = linalg_utils.einv(XtW.dot(X))
        P = W - XtW.T.dot(XtWX_inv).dot(XtW)
        # P = W - np.linalg.multi_dot([XtW.T, XtWX_inv, XtW])
        Py = P.dot(y)
        H = []
        for i, Ji in enumerate(jac_mats):
            for j, Jj in enumerate(jac_mats):
                if j >= i:
                    PJi = sps.coo_matrix.dot(Ji.T, P).T
                    PJj = sps.coo_matrix.dot(Jj.T, P).T
                    yPJi = sps.coo_matrix.dot(Ji.T, Py).T
                    JjPy = Jj.dot(Py)
                    Hij = -(PJi.dot(PJj)).diagonal().sum()\
                        + (2 * (yPJi.dot(P)).dot(JjPy))[0]
                    H.append(Hij[0])
        H = linalg_utils.invech(np.array(H))
        return H
    
class GLMM_PQL(GLMM):
    def __init__(self, fixed_effects, random_effects, yvar, data, fam,
                 error_structure=None, acov=None):
        
        self.f = fam
        self.fe, self.re, self.yvar = fixed_effects, random_effects, yvar
        self.error_struct, self.acov = error_structure, acov
        self.data = data
        self.mod = lmm.LMM(fixed_effects, random_effects, yvar, data, 
                            error_structure, acov)
        self.mod.fit()
        self.y = mod.y
        
    def fit(self, n_iters=200, tol=1e-3):
        
        mod = self.mod
        theta = mod.params
        for i in range(n_iters):
            eta = mod.predict()
            
            mu = self.f.inv_link(eta)
            v = mu * (1 - mu)
            gp = self.f.link.dlink(mu)
            nu = eta + gp*(y - mu)
            W = 1 / (v[:, 0] * (self.f.link.dlink(mu)**2)[:, 0])
            W = np.diag(1/np.sqrt(W))
            
            
            mod = GLMM(self.fe, self.re, self.yvar, self.data, W=W)
            mod.y = nu
            mod.fit()
            tvar = (np.linalg.norm(theta)+np.linalg.norm(mod.params))
            eps = np.linalg.norm(theta - mod.params) / tvar
            if eps < tol:
                break
            theta = mod.params
        self.mod = mod
        res = self.mod.optimizer
        self.params = res.x
        G, Ginv, SigA, R, Rinv, SigE = self.mod.params2mats(res.x)
        self.G, self.Ginv, self.R, self.Rinv = G, Ginv, R, Rinv
        self.SigA, self.SigE = SigA, SigE
        W = linalg_utils.woodbury_inversion(self.mod.Z, C=G, A=R)
        X = self.mod.X
        XtW = X.T.dot(W)
        self.optimizer = res
        self.hessian_est = self.mod.hessian(self.params)
        self.hessian_inv = np.linalg.pinv(self.hessian_est)
        self.SE_theta = np.sqrt(np.diag(self.hessian_inv))
        self.grd = self.mod.gradient(self.params)
        self.gnorm = np.linalg.norm(self.grd) / len(self.params)
        self.b = linalg_utils.einv(XtW.dot(X)).dot(XtW.dot(self.mod.y))
        self.SE_b = np.sqrt(np.diag(linalg_utils.einv(XtW.dot(X))))
        self.r = self.mod.y - self.mod.X.dot(self.b)
        self.u = G.dot(self.mod.Z.T.dot(W).dot(self.r))
        res = pd.DataFrame(np.concatenate([self.params[:, None], self.b]),
                           columns=['Parameter Estimate'])
        res['Standard Error'] = np.concatenate([self.SE_theta, self.SE_b])
        res['t value'] = res['Parameter Estimate'] / res['Standard Error']
        res['p value'] = sp.stats.t.sf(np.abs(res['t value']),
                                       X.shape[0]-len(self.params)) * 2.0
        res.index = self.mod.res_names
        self.res = res
        n_obs, k_params = self.mod.X.shape[0], len(self.params)
        
        self.ll = self.mod.loglike(self.params)
        self.aic = self.ll + (2 * k_params)
        self.aicc = self.ll + 2*k_params*n_obs / (n_obs - k_params - 1)
        self.bic = self.ll + k_params*np.log(n_obs)
        self.caic = self.ll + k_params * np.log(n_obs+1)
        self.r2_fe = 1 - np.var(self.mod.y - self.mod.X.dot(self.b)) / np.var(self.mod.y)
        self.r2_re = 1 - np.var(self.mod.y - self.mod.Z.dot(self.u)) / np.var(self.mod.y)
        self.r2 = 1 - np.var(self.mod.y - self.mod.predict()) / np.var(self.mod.y)
        self.sumstats = np.array([self.aic, self.aicc, self.bic, self.caic,
                                  self.r2_fe, self.r2_re, self.r2])
        self.sumstats = pd.DataFrame(self.sumstats, index=['AIC', 'AICC', 'BIC',
                                                           'CAIC', 
                                                           'FixedEffectsR2',
                                                           'RandomEffectsR2', 
                                                           'R2'])
        
        
        
        
        
   