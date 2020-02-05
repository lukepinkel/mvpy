#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:14:10 2019

@author: lukepinkel
"""

import numpy as np
import scipy as sp
import pandas as pd
from numpy import sqrt, ones, eye, zeros, dot, diag, kron
from numpy.linalg import (multi_dot, eig, eigh, svd, inv, pinv, cholesky, det, 
                          norm, LinAlgError)
from scipy.linalg import qr
import scipy.sparse as sps
import scipy.sparse.linalg as spl
from ..utils.base_utils import check_type, cov, csd

def _check_1d(x):
    if x.ndim>1:
        x = x.reshape(-1)
    return x

def _check_2d(x):
    if x.ndim!=2:
        x = x[:, None]
    return x

def _check_0d(x):
    if type(x) is np.ndarray:
        if x.ndim==1:   
            x = x[0]
        elif x.ndim==2:
            x = x[0][0]   
    return x


def _check_np(x):
    if ((type(x)==pd.DataFrame)|(type(x)==pd.Series)):
        x = x.values
    return x

def vec(X):
    '''
    Takes an n \times p matrix and returns a 1 dimensional np vector
    '''
    return X.reshape(-1, order='F')

def invec(x, n_rows, n_cols):
    '''
    Takes an np 1 dimensional vector and returns an n \times p matrix
    '''
    return x.reshape(int(n_rows), int(n_cols), order='F')

def vech(X):
    '''
    Half vectorization operator; returns an \frac{(n+1)\times n}{2} vector of
    the stacked columns of unique items in a symmetric  n\times n matrix
    '''
    rix, cix = np.triu_indices(len(X))
    res = X.T.take(rix*len(X)+cix)
    return res


def invech(v):
    '''
    Inverse half vectorization operator
    '''
    rows = int(np.round(.5 * (-1 + np.sqrt(1 + 8 * len(v)))))
    res = np.zeros((rows, rows))
    if v.dtype == 'complex128':
        res = res.astype(complex)
    res[np.triu_indices(rows)] = v
    res = res + res.T
    res[np.diag_indices(rows)] /= 2
    return res

def vechc(X):
    '''
    The same as the vech operator, only returns a 2 dimensional vector
    '''
    v = vech(X)
    if len(v.shape)==1:
        v = v[:, None]
    return v

def vecc(X):
    '''
    The same as the vecc operator, only returns a 2 dimensional vector
    '''
    v = vec(X)
    if len(v.shape)==1:
        v = v[:, None]
    return v

def commutation_matrix(p, q):
    '''
    Matrix K_{np} such that for an n\times p matrix X, vec(X^{T})=K vec(X)
    '''
    K = np.eye(p * q)
    #indices = np.arange(p * q).reshape((p, q), order='F')
    ix = np.arange(p * q).reshape((p, q), order='F').reshape(p*q, order='C')
    return K.take(ix, axis=0)

def duplication_matrix(n):
    '''
    Matrix D_{p} such that for a p\times p matrix X, D vech(X)= vec(X)
    '''
    D = [invech(x.astype(int)).ravel() for x in np.eye(n * (n + 1) // 2)]
    D = np.array(D).T
    return D

def psuedo_duplication_matrix(n):
    D = duplication_matrix(n)
    Dp = np.linalg.pinv(np.dot(D.T, D)).dot(D.T) #Redundant
    return Dp

def elimination_matrix(n):
    '''
    Matrix L_{p} such that for a p\times p matrix X, L vec(X)= vech(X)
    '''
    vech_indices = vec(np.tril(np.ones((n, n))))
    return np.eye(n * n)[vech_indices != 0]

def dmat(n):
    return duplication_matrix(n)

def dpmat(n):
    return psuedo_duplication_matrix(n)

def lmat(n):
    return elimination_matrix(n)

def kmat(p, q):
    return commutation_matrix(p, q)

def omat(p, q=None):
    if q is None:
        q = 1
    return zeros((p, q))

def jmat(p, q=None):
    if q is None:
        q = 1
    return ones((p, q))

def pre_post_elim(X):
    n, p = X.shape
    n, p = int(sqrt(n)), int(sqrt(p))
    Y = mdot([dmat(n).T, X, dmat(p)])
    return Y

def khatri_rao(X, Y):
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    p = X.shape[1]
    Res = np.stack([kron(X[:, i], Y[:, i]) for i in range(p)], axis=1)
    if (x_is_pd)|(y_is_pd):
        Res = pd.DataFrame(Res)
    return Res

def sdot(X, Y):
    return X.dot(Y)

def sparse_cholesky(A, permute='MMD_AT_PLUS_A'):
    if sps.issparse(A) is False:
        A = sps.csc_matrix(A)
    lu = spl.splu(A, permc_spec=permute)
    n = A.shape[0]
    Pr = sps.lil_matrix((n, n))
    Pc = sps.lil_matrix((n, n))
    
    Pr[lu.perm_r, np.arange(n)] = 1
    Pc[np.arange(n), lu.perm_c] = 1
    
    Pr = Pr.tocsc()
    Pc = Pc.tocsc()

    L, U = lu.L, lu.U
    L = Pr.T*L.dot(sps.diags(U.diagonal()**0.5))
    return L

def nmat(n):
    Nn = eye(n**2)+kmat(n, n)
    return Nn

def block_upper_diag(A, bshape):
    n, p = A.shape
    q, k = bshape
    
    O12 = omat(n, k)
    O21 = omat(q, p)
    O22 = omat(q, k)
    res = np.block([[A, O12], [O21, O22]])
    return res
    
def mat_rconj(A):
    return eye(A.shape[0]) - A    
 
def xprod(X, Y=None):
    if Y is None:
        Y = X
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    XTY = dot(X.T, Y)
    
    if ((x_is_pd)|(y_is_pd)):
        XTY = pd.DataFrame(XTY, index=xcols, columns=ycols)
        
    return XTY

def xiprod(X, Y=None):
    if Y is None:
        Y = X
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    XTY = pinv(dot(X.T, Y))
    
    if ((x_is_pd)|(y_is_pd)):
        XTY = pd.DataFrame(XTY, index=xcols, columns=ycols)
        
    return XTY

def lstq(X, Y):
    B = mdot([xiprod(X), (X.T), (Y)])
    return B
    
def lstq_pred(X, Y):
    B = lstq(X, Y)
    Yhat = X.dot(B)
    return Yhat

def mdot(vars_):
    return multi_dot(vars_)

def normalize_diag(X):
    X, cols, ix, is_pd = check_type(X)
    D = diag(sqrt(1.0/diag(X)))
    Xh = multi_dot([D, X, D])
    return Xh

def valid_overlap(X, Y=None):
    if Y is None:
        Y = X
        
    X, xcols, xix, x_is_pd = check_type(X)
    Y, ycols, yix, y_is_pd = check_type(Y)
    validX = (1 - np.isnan(X))
    validY = (1 - np.isnan(Y))
    valid = dot(validX.T, validY)
    if x_is_pd:
        valid = pd.DataFrame(valid, columns=xcols, index=ycols)
    return valid

def blockm(X, Y, Z):
    return np.block([[X, Y], [Y.T, Z]])

def blockwise_inv(A, B, C, D):
    invD = pinv(D)
    Ainv = pinv(A - multi_dot([B, invD, C]))
    Binv = multi_dot([-Ainv, B, invD])
    Cinv = multi_dot([invD, C, -Ainv])
    Dinv = pinv(D - multi_dot([C, pinv(A), B])) 
    return Ainv, Binv, Cinv, Dinv


def psuedo_kron(A, B):
    p, k = A.shape[1], B.shape[1]
    res = []
    op = ones((1, k))
    for i in range(p):
        res.append(A[:, [i]].dot(op) * B)
    return np.block(res)

def mpkron(Vars_):
    res = psuedo_kron(Vars_[0], Vars_[1])
    Vars_ = Vars_[1:]
    for x in Vars_:
        res = psuedo_kron(res, x)
    return res

def sweep(X, partition):
    X, cols, ix, is_pd = check_type(X)
    A = X[:partition, :partition]
    B = X[:partition, partition:]
    Bt = X[:partition, partition:]
    if not np.allclose(B - Bt, 0):
        raise ValueError('Not a symmetrical array')
    D = X[partition:, partition:]
    Ainv = pinv(A)
    X12 = Ainv.dot(B)
    Xswp = np.block([[Ainv, X12], [-X12.T, D - B.T.dot(X12)]])
    return Xswp
   
def spsweep(X, partition):
    A = X[:partition, :partition]
    B = X[:partition, partition:]
    D = X[partition:, partition:]
    Ainv = sps.csc_matrix(sps.linalg.inv(A))
    X12 = sps.csc_matrix(Ainv.dot(B))
    Xswp = sps.bmat([[Ainv, X12], [-X12.T, D - B.T.dot(X12)]])
    return Xswp

def adf_mat(Z):
    S = cov(Z)
    mu = Z.mean(axis=0)
    Y = _check_np(Z)
    s = vech(_check_np(S))
    ss = [vech((Y[i] - mu)[:, None].dot((Y[i]-mu)[:, None].T)) 
          for i in range(Y.shape[0])]
    Gadf = np.sum([(si-s)[:, None].dot((si-s)[:, None].T) for si in ss],
                   axis=0)/Y.shape[0]
    return Gadf

def qcov(X, Y=None): 
    if np.isnan(X).any():
        X = np.ma.masked_invalid(X) 
    if Y is None:
        S = np.cov(X, rowvar=False)
    elif Y is not None:
        if np.isnan(Y).any():
            Y = np.ma.masked_invalid(Y)
        S = np.cov(X, Y, rowvar=False)[X.shape[1]:, :Y.shape[1]]
    return S
     
  
          
def kronvec_mat(A_dims, B_dims, sparse=False):
  '''
  The K_{v} matrix allows for the vecorization of a kronecker product to be
  written as a kronecker product of the vecs such that for A_{n\times q} and
  B_{p\times r}:
    vec(A \otimes B) = K_{v}(vec(A)\otimes vec(B))
  
  Parameters
  -----------
  A_dims: list or tuple
    dimensions of A
  B_dims: list or tuple
    dimensions of B
  '''
  n, p = A_dims
  q, r = B_dims
  if sparse is False:
    Kv = kron(kron(eye(p), kmat(r, n)), eye(q))
  else:
    Kv = sps.kron(sps.eye(p), sparse_kmat(r, n))
    Kv = sps.kron(Kv, sps.eye(q))
  return Kv

def einv(A):
  try:
    A_inv = inv(A)
  except LinAlgError:
    A_inv = pinv(A)
  return A_inv            
    
    
      
def sparse_kmat(p, q):
  K = sps.eye(p * q)
  indices = np.arange(p * q).reshape((p, q), order='F')
  K = sps.csc_matrix(K)[indices.ravel()]
  return K  
                
def swp(g, k):
    '''
    Gaussian sweep operator, obtained from
    https://stackoverflow.com/questions/15767435/python-implementation-of-statistical-sweep-operator
    '''
    g = np.asarray(g)
    n = g.shape[0]
    if g.shape != (n, n):
        raise ValueError('Not a square array')
    if not np.allclose(g - g.T, 0):
        raise ValueError('Not a symmetrical array')
    if k >= n:
        raise ValueError('Not a valid row number')
    #  Fill with the general formula
    h = g - np.outer(g[:, k], g[k, :]) / g[k, k]
    # h = g - g[:, k:k+1] * g[k, :] / g[k, k]
    # Modify the k-th row and column
    h[:, k] = g[:, k] / g[k, k]
    h[k, :] = h[:, k]
    # Modify the pivot
    h[k, k] = -1 / g[k, k]
    return h


def fprime(Func, X, *args, eps=None):
    '''
    Derivative of a matrix function of a matrix using Magnus and Neudecker
    definition of a derivative as \frac{d vec(F(X))} {d vec(X)^{T}}
    
    Parameters:
        F: Matrix function
        X: Matrix to evaluate the derivative of F at
        args: Args for F
    
    Returns:
        J: Matrix of first derivatives
    
    '''
    if type(X)==float:
        def F(X, *args):
            X = np.array([X])
            return Func(*((X[:, None],) + args))
    elif X.ndim==1:
        def F(X, *args):
            X, _, _, _ = check_type(X)
            return Func(*((X[:, None],) + args))
    else:
        def F(X, *args):
            return Func(*((X,) + args))
    Y = F(*((X,)+args))
    x, y = vec(X), vec(Y)
    p, q = len(x), len(y)
    J = omat(q, p)
    ei = zeros(p)
    if eps is None:
        eps = (np.finfo(0.1).eps)**(1/3)*x+np.finfo(0.1).eps
    for i in range(p):
        ei[i] = eps[i]/2.0
        if X.ndim==1:
            x1 = vec(F(*((invec(x+ei, *X[:, None].shape),)+args)))
            x2 = vec(F(*((invec(x-ei, *X[:, None].shape),)+args)))
        else:
            x1 = vec(F(*((invec(x+ei, *X.shape),)+args)))
            x2 = vec(F(*((invec(x-ei, *X.shape),)+args)))
        J[:, i] = (x1 - x2) / (eps[i])
        ei[i] = 0
    return J


def fprime_cs(Func, X, *args, eps=None):
    '''
    Derivative of a matrix function of a matrix using Magnus and Neudecker
    definition of a derivative as \frac{d vec(F(X))} {d vec(X)^{T}}
    
    Parameters:
        F: Matrix function
        X: Matrix to evaluate the derivative of F at
        args: Args for F
    
    Returns:
        J: Matrix of first derivatives
    
    '''
    if type(X)==float:
        def F(X, *args):
            X = np.array([X])
            return Func(*((X[:, None],) + args))
    elif X.ndim==1:
        def F(X, *args):
            X, _, _, _ = check_type(X)
            return Func(*((X[:, None],) + args))
    else:
        def F(X, *args):
            return Func(*((X,) + args))
    Y = F(*((X,)+args))
    x, y = vec(X), vec(Y)
    p, q = len(x), len(y)
    J = omat(q, p)
    ei = zeros(p).astype(complex)
    if eps is None:
        eps = (np.finfo(0.1).eps)**(1.0/3.0)*x+np.finfo(0.1).eps
    for i in range(p):
        ei[i] = eps[i]*1.0j
        if X.ndim==1:
            x1 = vec(F(*((invec(x+ei, *X[:, None].shape),)+args)))
        else:
            x1 = vec(F(*((invec(x+ei, *X.shape),)+args)))
        J[:, i] = (x1.imag) / (eps[i])
        ei[i] = 0.0

    return J

def hess_approx(Func, X, *args, eps=None):
    if type(X)==float:
        def F(X, *args):
            X = np.array([X])
            return Func(*((X[:, None],) + args))
    elif X.ndim==1:
        def F(X, *args):
            X, _, _, _ = check_type(X)
            return Func(*((X[:, None],) + args))
    else:
        def F(X, *args):
            return Func(*((X,) + args))
    Y = F(*((X,)+args))
    x, y = vec(X), vec(Y)
    p, q = len(x), len(y)
    H = omat(q*p, p*q)
    ei = zeros(p)
    if eps is None:
        eps = (np.finfo(0.1).eps)**(1/3)*x+np.finfo(0.1).eps    
    
    for i in range(p):
        ei[i] = eps[i]/2.0
        x1 = vec(fprime(F, invec(x+ei, *X.shape), args=args))
        x2 = vec(fprime(F, invec(x-ei, *X.shape), args=args))
        H[:, i] = (x1 - x2) / (eps[i])
        ei[i] = 0
    return H



def diag2(X):
    '''
    Returns a matrix Y that has the diagonal values of input matrix X
    on its diagonals and zeros elsewhere
    
    Parameters:
        X: Matrix whose diagonals are to be used to construct Y
    
    Returns:
        Y: Matrix with the diagonals of X on its diagonals, and zeros elsewhere
    '''
    Y = diag(diag(X))
    return Y


def svd2(X):
    '''
    SVD that returns singular values in a diagonal matrix and V,
    not V^{T}
    
    Parameters:
        X: n by p matrix to subject to singular value decomposition
    
    Results:
        
        U: Left eigenvectors 
        S: Diagonal matrix of singular values
        V: Right eigenvectors
        
    '''
    U, s, Vt = svd(X, 0)
    S = diag(s)
    V = Vt.T
    return U, S, V



def sorted_eig(X):
    '''
    Sorted eigendecomposition
    
    Parameters:
        X: n by n matrix to be decomposed
    
    Returns:
        u: n vector of eigenvalues sorted by size
        V: n by n matrix of corresponding eigenvalues
    '''
    u, V = eig(X)
    idx = u.argsort()[::-1]
    u, V = u[idx], V[:, idx]
    return u, V

def sorted_eigh(X, return_idx=False):
    '''
    Sorted Hermitian Eigendecomposition
    
    Parameters:
        X: n by n matrix to be decomposed
        
    Returns:
        u: n vector of eigenvalues sorted by size
        V: n by n matrix of eigenvectors
        idx: If return_idx==True, returns the index of eigenvalues by size
    '''
    u, V = eigh(X)
    idx = u.argsort()[::-1]
    u, V = u[idx], V[:, idx]
    if return_idx is False:    
        return u, V
    elif return_idx is True:
        return u, V, idx

def inv_sqrt(X):
    '''
    Inverse matrix square root
    If X is not positive semidefinite then the eigenvalues are limited to 1e-12
    
    Parameters:
        X: Square symmetric matrix 
    
    Returns:
        Xisq: Square symmetric inverse square root of X 
    '''
    u, V = eig(X)
    U = diag(1 / sqrt(np.maximum(u, 1e-12)))
    Xisq = multi_dot([V, U, V.T])
    return Xisq


def inv_sqrth(X):
    '''
    Inverse matrix square root with eigh
    If X is not positive semidefinite then the eigenvalues are limited to 1e-12
    
    Parameters:
        X: Square symmetric matrix 
    
    Returns:
        Xisq: Square symmetric inverse square root of X 
    '''
    u, V = eigh(X)
    U = diag(1 / sqrt(np.maximum(u, 1e-12)))
    Xisq = multi_dot([V, U, V.T])
    return Xisq
    
def zca(X, S=None):
    '''
    Matrix whitening with PCA
    
    Parameters:
        X: n by m matrix to be whitened
        S: optional covariance matrix of X that is computed if not provided
    
    Returns:
        W: Whitening matrix for X
    '''
    if S is None:
        S = cov(X)
    W = inv_sqrt(S)
    return W

def wpca(X, S=None):
    '''
    Modified matrix whitening with partial reconstruction.
    Instead of reconstructing X = V(U^-1/2)V', X=(U^-1/2)V'
    
    Parameters:
        X: n by m matrix to be whitened
        S: optional covariance matrix of X that is computed if not provided
    
    Returns:
        W: Whitening matrix for X
    '''
    if S is None:
        S = cov(X)
    u, V = eig(S)
    U = diag(1 / sqrt(u))
    W = dot(U, V.T)
    return W

def cholesky_whitening(X, S=None):
    '''
    Whitening with cholesky decomposition
    
    Parameters:
        X: n by m matrix to be whitened
        S: optional covariance matrix of X
    
    Returns:
        W: Whitening matrix for X
    '''
    if S is None:
        S = cov(X)
    W = cholesky(pinv(S)).T
    return W

def whiten(X, method='PCA'):
    '''
    Whitening
    
    Parameters:
        X: n by m matrix to be whitened
        method: method used to whiten X
    
    Returns:
        Z: whitened version of X
    '''
    if method=='PCA':
        W = wpca(X)
    if method=='ZCA':
        W = zca(X)
    if method=='chol':
        W = cholesky_whitening(X)
    Z = dot(W, X.T).T
    return Z

def eig_adjust(X):
    '''
    Approximation of X that is positive semidefinite through reconstructing
    X with eigenvalues less than 1e-9 being set to 1e-9
    
    Parameters:
        X: non positive semidefinite square symmetric matrix
        
    Returns:
        Xh: approximation of X that is positive semidefinite
    '''
    u, V = eig(X)
    u = np.maximum(u, 1e-9)
    
    Xh = multi_dot([V, diag(u), V.T])
    k = diag(1 / sqrt(diag(Xh)))
    Xh = multi_dot([k, Xh, k])
    return Xh
    
def ridge_adjust(X, lower_limit=0):
    '''
    Approximation of X that is non singular through scaling off diagonals
    
    Parameters:
        X: non positive semidefinite square symmetric matrix
        
    Returns:
        Xh: approximation of X that is positive semidefinite
    '''
    n = len(X)
    while det(X)<lower_limit:
        X = X + np.eye(n) / 100
        
    k = diag(1 / sqrt(diag(X)))
    Xh = multi_dot([k, X, k])
    return Xh

def near_psd(X, method='eig', lower_limit=0):
    '''
    The nearest positive semidefinite version of X
    
    Parameters:
        X: Square symmetric matrix to be made positive semidefinite
        method: method of adjustment, either eig for reconstruction from 
        eigenvalues that are at least 1e-9, or ridge for scaling off diagonals
    
    Returns:
        Xh: nearest positive semidefinite approximation of X
    '''
    X, cols, ix, is_pd = check_type(X)
    if method=='eig':
        Xh = eig_adjust(X)
    elif method=='ridge':
        Xh = ridge_adjust(X, lower_limit=lower_limit)
    if is_pd is True:
        Xh = pd.DataFrame(Xh, index=ix, columns=cols)
    return Xh
    

def multi_rand(R, size=1000):
    '''
    Generates multivariate random normal matrix
    
    Parameters:
        R: n by n covariance or correlation matrix of the distribution from 
        which the random numbers are to be pulled
        size: size of the random sample
    
    Returns:
        Y: size by n matrix of multivariate random normal values
    '''
    R, col, ix, is_pd = check_type(R)
    
    n = R.shape[0]
    X = csd(whiten(csd(np.random.normal(size=(size, n)))))
    
    W = chol(R)
    Y = X.dot(W.T)
    if is_pd:
        Y = pd.DataFrame(Y, columns=col)
    return Y





def multi_corr(R):
    '''
    Returns the multiple correlation of each feature in the provided 
    correlation matrix with all the other features
    
    Parameters:
        R: Correlation matrix
    
    Returns:
        rx: The multiple correlation of each feature with all the other
            features
    '''
    rx = 1 - 1 / diag(inv(R))
    return rx

def VgQ_Ortho(L, gamma):
    '''
    VgQ subroutine for orthogonal rotation
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    I, C = eye(p), ones((p, p))/p
    H = multi_dot([(I - gamma*C), L2])
    ft = np.sum(L2*H) * 0.25
    Gq = L * H
    return -ft, -Gq

    
def VgQ_Obli(L, gamma):
    '''
    VgQ subroutine for oblique rotation
    
    Parameters:
        L: Loadings matrix of p features and q factors
        gamma: Coefficient that determines the type of rotation
    
    Returns:
        ft: Criteria function
        Gq: Gradient of the function at L
    '''
    p, q = L.shape
    L2 = L**2
    I, C, N = eye(p), ones((p, p))/p, ones((q, q)) - eye(q)
    H = multi_dot([(I - gamma*C), L2, N])
    ft = np.sum(L2*H) * 0.25
    Gq = L * H
    return ft, Gq

def rotate_ortho(A, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=1000):
    '''
    Orthogonal rotation
    
    Parameters:
        A: Loadings matrix
        T: Initial rotation matrix
        alpha: Coefficient that determines the step size
        gamma: Coefficient that determines the type of rotation
        tol: Tolerance that determines convergance
        n_iters: The maximum number of iterations before stopping
    
    Returns:
        T: Rotation matrix
    '''
    if T is None:
        T = np.eye(A.shape[1])
    L = dot(A, T)
    ft, Gq = VgQ_Ortho(L, gamma)
    G = dot(A.T, Gq)
    for i in range(n_iters):
        M = dot(T.T, G)
        S = (M + M.T) / 2.0
        Gp = G - dot(T, S)
        s = norm(Gp)
        if s<tol:
            break
        alpha = 2.0 * alpha
        for c in range(10):
            X = T - alpha * Gp
            U, D, V = svd(X, full_matrices=False)
            Tt = dot(U, V)
            L = dot(A, Tt)
            ft_new, Gq = VgQ_Ortho(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G=dot(A.T, Gq)
    return T


def rotate_obli(A, T=None, alpha=1.0, gamma=0, tol=1e-9, n_iters=500):
    '''
    Oblique rotation
    
    Parameters:
        A: Loadings matrix
        T: Initial rotation matrix
        alpha: Coefficient that determines the step size
        gamma: Coefficient that determines the type of rotation
        tol: Tolerance that determines convergance
        n_iters: The maximum number of iterations before stopping
    
    Returns:
        T: Rotation matrix
    '''
    if T is None:
        T = np.eye(A.shape[1])
    Tinv = inv(T)
    L = dot(A, Tinv.T)
    ft, Gq = VgQ_Obli(L, gamma)
    G = -multi_dot([L.T, Gq, Tinv]).T
    for i in range(n_iters):
        TG = T*G
        Gp = G - dot(T, diag(np.sum(TG, axis=0)))
        s = norm(Gp)
        if s<tol:
            break
        alpha = 2.0 * alpha
        for c in range(10):
            X = T - alpha * Gp
            X2 = X**2
            V = diag(1 / sqrt(np.sum(X2, axis=0)))
            Tt = dot(X, V)
            Tinv = pinv(Tt)
            L = dot(A, Tinv.T)
            ft_new, Gq = VgQ_Obli(L, gamma)
            if ft_new < (ft - 0.5*s**2*alpha):
                break
            else:
                alpha = alpha * 0.5
        ft, T =ft_new, Tt
        G = -multi_dot([L.T, Gq, Tinv]).T
    return T

def rotate(A, method, T=None, tol=1e-9, alpha=1.0,
           n_iters=500, custom_gamma=None, k=4):
    '''
    Rotation of loadings matrix
    
    Parameters:
        A: Loadings Matrix
        method: Type of rotation
        T: Initial rotation matrix
        tol: Tolerance controlling convergance
        alpha: Parameter controlling step size taken in GPA algorithm
        n_iters: Maximum number of iterations before convergance
        custom_gamma: Coefficient used to customize non standard oblique rotations
    
    Returns:
        L: Rotated loadings matrix
        T: Rotation matrix
    
    Methods are:
        quartimax
        biquartimax
        varimax
        equamax
        quartimin
        biquartimin
        covarimin
        oblique
    
    '''
    if type(A) is pd.DataFrame:
        ix, cols = A.index, A.columns
    if method == 'quartimax':
        gamma = 0
        rotation_type = 'orthogonal'
    if method == 'biquartimax':
        gamma = 0.5
        rotation_type = 'orthogonal'
    if method == 'varimax':
        gamma = 1.0
        rotation_type = 'orthogonal'
    if method == 'equamax':
        gamma = A.shape[0] / 2
        rotation_type = 'orthogonal'
    if method == 'promax':
        gamma = 1.0
        rotation_type = 'orthogonal'
    if method == 'quartimin':
        gamma = 0.0
        rotation_type = 'oblique'
    if method == 'biquartimin':
        gamma = 0.5
        rotation_type = 'oblique'
    if method == 'covarimin':
        gamma = 1.0
        rotation_type = 'oblique'
    if method == 'oblique':
        if custom_gamma is None:
            gamma = -0.1
        else:
            gamma = custom_gamma
        rotation_type = 'oblique'
        
    if rotation_type == 'orthogonal':
        T = rotate_ortho(A, T=T, alpha=alpha, gamma=gamma, tol=tol, n_iters=n_iters)
        L = dot(A, T)
        if method == 'promax':
            H = abs(L)**k/L
            V = multi_dot([pinv(dot(A.T, A)), A.T, H])
            D = diag(sqrt(diag(inv(dot(V.T, V)))))
            T = inv(dot(V, D)).T
            L = dot(A, T)
            
    elif rotation_type == 'oblique':
        T = rotate_obli(A, T=T, alpha=alpha, gamma=gamma, tol=tol, n_iters=n_iters)
        L = dot(A, inv(T).T)
    if type(A) is pd.DataFrame:
        L = pd.DataFrame(L, index=ix, columns=cols)
    return L, T

def replace_diagonal(X, h):
    '''
    Replace diagonals with custom vector.  Utility tool used for PAF
    
    Parameters:
        X: Square matrix whose diagonals are to be replaced
        h: Vector of compatible dimensions with X
    
    Returns:
        Xh: X with diagonals replaced by h
    '''
    Xh = X - diag2(X) + diag(h)
    return Xh



        
        
def woodbury_inversion(Umat, Vmat=None, C=None, Cinv=None, A=None, Ainv=None):
    if Ainv is None:
        Ainv = einv(A)
    if Cinv is None:
        Cinv = einv(C)
    if Vmat is None:
        Vmat = Umat.T
    T = Ainv.dot(Umat)
    H = np.linalg.inv(Cinv + Vmat.dot(T))
    W = Ainv - np.linalg.multi_dot([T, H, Vmat, Ainv])
    return W
    
def chol(A):
  try:
    L = cholesky(A)
  except LinAlgError:
    #print('non positive semidefinite augmented mme matrix')
    u, V = eigh(A)
    U = diag(sqrt(np.maximum(u, 0)))
    Q, R = qr((V.dot(U)).T)
    L = R.T
  return L
    
def symm_deriv(X, a, b):
    L = lmat(a)
    return L.dot(X).dot(L.T)
    

def confound_adjust(Y, Confounds):
    '''
    Returns Y adjusted for counfounds G
    Y_adj = Y - G(G'G)^{-1}G'Y
    
    Parameters:
        Y: Matrix or vector
        Confounds: Confounds to be regressed out of Y
    
    Returns:
        Ya: Y adjusted for confounds
    '''
    Y, ycols, yix, y_is_pd = check_type(Y)
    G, gcols, gix, g_is_pd = check_type(Confounds)
    
    Ya = Y - multi_dot([G, inv(dot(G.T, G)), G.T, Y])
    if y_is_pd:
        Ya = pd.DataFrame(Ya, columns=ycols, index=yix)
    return Ya


def normdiff(a, b):
    diff = np.linalg.norm(a - b)
    diff /= (np.linalg.norm(a) + np.linalg.norm(b))
    return diff


def fastls(X, y):
    '''
    Fast albeit potentially numerically inaccurate algorithm to compute
    OLS coefficients, sum of square errors, and the covariance matrix for
    the coefficient estimates (given a correctly specified model)
    '''
    n, p = X.shape
    G = X.T.dot(X)
    c = X.T.dot(y)
    L = chol(G)
    w = np.linalg.solve(L, c)
    s2 =  (np.dot(y.T, y) - w.T.dot(w)) / (n - p)
    beta = np.linalg.solve(L.T, w)
    Linv = np.linalg.inv(L)
    Ginv = np.dot(Linv.T, Linv)
    beta_cov = s2 * Ginv
    return s2, beta, beta_cov
    


def add_chol_row(xnew, xold, L=None):
    xtx = xnew
    norm_xnew = np.sqrt(xtx)
    if L is None:
        L = np.atleast_2d(norm_xnew)
        return L
    else:
        Xtx = xold
        r = sp.linalg.solve(L, Xtx)
        rpp = np.sqrt(xtx - np.sum(r**2))
        A = np.block([[L, np.zeros((L.shape[0], 1))],
                       [r, np.atleast_1d(rpp)]])
        return A


