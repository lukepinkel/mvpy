#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:03:34 2019

@author: lukepinkel
"""
import pandas as pd
import numpy as np
import scipy.sparse as sps


from collections import OrderedDict
from numpy import eye, kron, sqrt, zeros, diag
from patsy import dmatrix
from numpy.linalg import inv, pinv, slogdet
from scipy.linalg import block_diag
from scipy.optimize import minimize
from ..utils.data_utils import dummy_encode
from ..utils.linalg_utils import (khatri_rao, vech, invech, vecc, dmat, kronvec_mat,
                            _check_np, chol, _check_1d, woodbury_inversion,
                            einv, _check_2d, vec)


class LMM:
  
  def __init__(self, fixed_effects, random_effects,  yvar, data, 
               error_structure=None, acov=None):
    '''
    Linear Mixed Model
    
    Parameters
    ----------
    fixed_effects: str
      String formula containing the column names corresponding to the 
      fixed effect terms.  This is to be evaluated by patsy
      
    random_effects: dict
      Dictionary whose keys correspond to factors, and whose values are
      formulas specifying the random effect terms.  The keys are evaluated
      as categorical and dummy encoded.
    
    y: str or list
      If y is a string, then a univariate linear mixed model is fit, but if
      y is a list of m strings, then an m-variate linear mixed model is fit.  
      It is assumed that all fixed and random effects are identical for all
      dependent variables.
    
    data: DataFrame
      Pandas DataFrame containing n_obs by n_features including the relavent
      terms to the model
    
    error_structure: default None, str
      Error structure defaults to iid, but a factor level may be provided via
      a string referencing a column name, which will then be used to construct
      the error covariance.  Implemented for multivariate linear models, where
      id is repeated across the multiple dependent variables, and has the
      structure Cov(Error) = V_{m\times m}\otimes I_{n}
    '''
    
    n_obs = data.shape[0]
    X = dmatrix(fixed_effects, data=data, return_type='dataframe')
    
    Z = []
    re_struct = OrderedDict()
    if type(yvar) is list:
      n_vars = len(yvar)
    else:
      n_vars = 1
      
    for key in random_effects.keys():
      Ji = dummy_encode(data[key], complete=True)
      Zij = dmatrix(random_effects[key], data=data, return_type='dataframe')
      Zi = khatri_rao(Ji.T, Zij.T).T
      Z.append(Zi)
      k = Zij.shape[1]*n_vars
      if (acov is not None):
          if acov[key] is not None:
               acov_i = acov[key]
          else:
              acov_i = eye(Ji.shape[1])
      else:
          acov_i = eye(Ji.shape[1])
      re_struct[key] = {'n_units':Ji.shape[1],
                        'n_level_effects':Zij.shape[1],
                        'cov_re_dims':k,
                        'n_params':((k + 1.0) * k) / 2.0,
                        'vcov':eye(k),
                        'params':vech(eye(k)),
                        'acov':acov_i}
        
    Z = np.concatenate(Z, axis=1)
    
    error_struct = OrderedDict()
    error_struct['vcov'] = eye(n_vars)
    error_struct['acov'] = eye(n_obs)
    error_struct['params'] = vech(eye(n_vars))
    
    if type(yvar) is str:
      y = data[[yvar]]
      
    elif type(yvar) is list:
      y = vecc(data[yvar].values)
      Z = kron(eye(n_vars), Z)
      X = np.vstack([X for i in range(n_vars)])
      
    var_params = np.concatenate([re_struct[key]["params"] for key in re_struct.keys()])
    err_params = error_struct['params']
    partitions = [0]+[re_struct[key]['n_params'] for key in re_struct.keys()]
    partitions += [len(error_struct['params'])]
    theta = np.concatenate([var_params, err_params])
    partitions2 = [0]+[re_struct[key]['n_units']*re_struct[key]['cov_re_dims'] for key in re_struct.keys()]
    partitions2 = np.cumsum(partitions2)
    var_struct = OrderedDict()
    for key in re_struct.keys():
      var_struct[key] = [re_struct[key]['vcov'].shape, re_struct[key]['acov']]
    var_struct['error'] = [error_struct['vcov'].shape, error_struct['acov']]
    
    Zs = OrderedDict()
    ZoZ = OrderedDict()
    for i in range(len(re_struct)):
      key = list(re_struct)[i]
      Zs[key] = sps.csc_matrix(Z[:, partitions2[i]:partitions2[i+1]])
      ZoZ[key] = sps.csc_matrix(sps.kron(Zs[key], Zs[key]))
    
    deriv_mats = OrderedDict()
    for key in var_struct.keys():
      Sv_shape, Av = var_struct[key]
      Av_shape = Av.shape
      Kv = kronvec_mat(Sv_shape, Av_shape, sparse=True)
      Ip = sps.csc_matrix(sps.eye(np.product(Sv_shape)))
      vecAv = sps.csc_matrix(vecc(Av))
      
      D =  sps.csc_matrix(Kv.dot(sps.kron(Ip, vecAv)))
      if key!='error':
          D = sps.csc_matrix(ZoZ[key].dot(D))
      deriv_mats[key] = D.dot(sps.csc_matrix(dmat(int(sqrt(D.shape[1])))))
    
    self.var_struct = var_struct
    self.deriv_mats = deriv_mats
    self.bounds = [(0, None) if x==1 else (None, None) for x in theta]
    self.theta = theta
    self.partitions = np.cumsum(partitions)
    J = sps.hstack([deriv_mats[key] for key in deriv_mats])
    self.jac_mats = [J[:, i].reshape(Z.shape[0], Z.shape[0], order='F') for i in range(J.shape[1])]
    
    self.X, self.Z, self.y = _check_np(X), _check_np(Z), _check_np(y)
    self.error_struct = error_struct
    self.re_struct = re_struct
    self.ZoZ = ZoZ
    
    
    self.n_vars = n_vars
    self.XZY = np.block([X, Z, y])
    self.XZ = np.block([X, Z])
    self.A = np.block([[X, Z], [zeros((Z.shape[1], X.shape[1])),
                        eye(Z.shape[1])]])
  
    
    
    
    
  def params2mats(self, theta=None):
    '''
    Create variance matrices from parameter vector
    Parameters
    ------------
    theta: array
      Vector containing relavent model terms
    '''
    if theta is None:
      theta=self.theta
    partitions=self.partitions
    error_struct = self.error_struct
    re_struct = self.re_struct
    
    Glist, Ginvlist, SigA = [], [], []
    for i, key in enumerate(re_struct.keys()):
      a, b = int(partitions[i]), int(partitions[i+1])
      Vi = invech(theta[a:b])
      Ai = re_struct[key]['acov']
      Glist.append(kron(Vi, Ai))
      Ginvlist.append(kron(pinv(Vi), Ai))
      SigA.append(Vi)
    Verr = invech(theta[int(partitions[-2]):int(partitions[-1])])
    R = kron(Verr, error_struct['acov'])
    Rinv = kron(inv(Verr), error_struct['acov'])
    G, Ginv = block_diag(*Glist), block_diag(*Ginvlist)
    
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
    yRy = _check_np(t).dot(y)
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
    theta = _check_1d(theta)
    G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
    re_struct, error_struct = self.re_struct, self.error_struct
    C = self.mmec(Rinv, Ginv)
    M = self.mme_aug(Rinv, Ginv, C=C)
    L = chol(M)
    logdetC = 2*np.sum(np.log(diag(L)[:-1]))
    yPy = L[-1, -1]**2
    logdetG = 0.0
    for key, Vi in list(zip(re_struct.keys(), SigA)):
      logdetG += re_struct[key]['n_units']*slogdet(Vi)[1]
    logdetR = error_struct['acov'].shape[0]*slogdet(SigE)[1]
    LL = logdetR+logdetC + logdetG + yPy
    return LL
    
    
    
  def fit(self, optimizer_kwargs={}, maxiter=100, verbose=2, hess_opt=False):
    if hess_opt is False:
      res = minimize(self.loglike, self.theta, bounds=self.bounds, 
               options={'verbose':verbose, 'maxiter':maxiter}, method='trust-constr', 
               jac=self.gradient, **optimizer_kwargs)
    else:
      res = minimize(self.loglike, self.theta, bounds=self.bounds, 
               options={'verbose':verbose, 'maxiter':maxiter}, method='trust-constr', 
               jac=self.gradient, hess=self.hessian, **optimizer_kwargs)
      
    self.params = res.x
    G, Ginv, SigA, R, Rinv, SigE = self.params2mats(res.x)
    self.G, self.Ginv, self.R, self.Rinv = G, Ginv, R, Rinv
    self.SigA, self.SigE = SigA, SigE
    W = woodbury_inversion(self.Z, C=G, A=R)
    X = self.X
    self.optimizer = res
    self.hessian_est = self.hessian(self.params)
    self.hessian_inv = pinv(self.hessian_est)
    self.SE_theta = sqrt(diag(self.hessian_inv))
    self.gnorm = np.linalg.norm(self.gradient(self.params))/len(self.params)
    self.b = einv(X.T.dot(W).dot(X)).dot(X.T.dot(W).dot(self.y))
    self.SE_b = sqrt(diag(einv(X.T.dot(W).dot(X))))
    self.r = self.y - self.X.dot(self.b)
    self.u = G.dot(self.Z.T.dot(W).dot(self.r))
    res = pd.DataFrame(np.concatenate([self.params[:, None], self.b]), 
                       columns=['Parameter Estimate'])
    res['Standard Error'] = np.concatenate([self.SE_theta, self.SE_b])
    self.res = res
    
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
    
    \partial\mathcal{L}=vec(Py)'\partial V-(vec(Py)\otimes vec(Py))'\partial V
    
    Parameters
    ----------
    theta: array
      Vector of parameters
    
    Returns
    --------
    g: array
      gradient vector of one dimensions (for compatibility with scipy minimize)
    
    
    '''
    theta = _check_1d(theta)
    G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
    deriv_mats = self.deriv_mats
    X, Z, y = self.X, self.Z, self.y
    W = woodbury_inversion(Z, C=G, A=R)
    XtW = X.T.dot(W)
    XtWX_inv = einv(XtW.dot(X))
    P = W - XtW.T.dot(XtWX_inv).dot(XtW)
    dP = P.reshape(np.product(P.shape), 1, order='F')
    Py = P.dot(y)
    PyPy = kron(Py, Py)
    #PyPy = vec(_check_2d(Py).dot(_check_2d(Py).T))[:, None] effecient only at large heterogenous n
    g = []
    for key in deriv_mats.keys():
        JF_Omega = deriv_mats[key]
        g_i = JF_Omega.T.dot(dP) - JF_Omega.T.dot(PyPy)
        g.append(g_i)
    g = np.concatenate(g)
    return _check_1d(g)
  
  def hessian(self, theta):
    theta = _check_1d(theta)
    G, Ginv, SigA, R, Rinv, SigE = self.params2mats(theta)
    jac_mats = self.jac_mats
    X, Z, y = self.X, self.Z, self.y
    W = woodbury_inversion(Z, C=G, A=R)
    XtW = X.T.dot(W)
    XtWX_inv = einv(XtW.dot(X))
    P = W - XtW.T.dot(XtWX_inv).dot(XtW)
    #P = W - np.linalg.multi_dot([XtW.T, XtWX_inv, XtW])
    Py = P.dot(y)
    H = []
    for i, Ji in enumerate(jac_mats):
      for j, Jj in enumerate(jac_mats):
        if j>=i:
            PJi, PJj = sps.coo_matrix.dot(Ji.T, P).T, sps.coo_matrix.dot(Jj.T, P).T
            yPJi = sps.coo_matrix.dot(Ji.T, Py).T
            JjPy = Jj.dot(Py)
            Hij = -(PJi.dot(PJj)).diagonal().sum()+(2*(yPJi.dot(P)).dot(JjPy))[0]
            H.append(Hij[0])
    H = invech(np.array(H))
    return H





