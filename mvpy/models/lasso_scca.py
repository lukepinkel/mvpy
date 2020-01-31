#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:41:52 2020

@author: lukepinkel
"""
import time # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import pandas as pd # analysis:ignore
import scipy.optimize as opt# analysis:ignore

from ..utils import linalg_utils, base_utils # analysis:ignore

import matplotlib.pyplot as plt # analysis:ignore

def sft(x, t):
    y = np.maximum(np.abs(x) - t, 0) * np.sign(x)
    return y

def left_update(S, w2, delta):
    w1_new = sft(S.dot(w2), delta)
    w1_new/= np.linalg.norm(w1_new)
    return w1_new
    

def right_update(S, w1, delta):
    w2_new = sft(w1.dot(S), delta)
    w2_new/= np.linalg.norm(w2_new)
    return w2_new

def lfunc(delta, S, w2, c1):
    if np.max(np.abs(S.dot(w2)) - delta)<=0:
        return -1
    else:
        return np.linalg.norm(left_update(S, w2, delta), 1) - c1
    
def rfunc(delta, S, w1, c2):
    if np.max(np.abs(w1.dot(S)) - delta)<=0:
        return -1
    else:
        return np.linalg.norm(right_update(S, w1, delta), 1) - c2


def weight_update(S, w2, c1, c2):
    delta1, delta2 = 0, 0
    w1_new = left_update(S, w2, delta1)
    if np.linalg.norm(w1_new, 1)>c1:
        delta1 = opt.root_scalar(lfunc, bracket=[0, 50], args=(S, w2, c1), method='brentq')                    
        w1_new = left_update(S, w2, delta1.root)
    w1 = w1_new
    
    w2_new = right_update(S, w1, delta2)
    if np.linalg.norm(w2_new, 1)>c2:
        delta2 = opt.root_scalar(rfunc, bracket=[0, 50], args=(S, w1, c2), method='brentq')                   
        w2_new = right_update(S, w1, delta2.root)
    w2 = w2_new
    return w1, w2
        

def pobjfunc(w1, w2, S, c1, c2):
    ssq = w1.T.dot(S).dot(w2)
    pf = ssq - np.linalg.norm(w1, 1) - np.linalg.norm(w2, 2)
    return pf
    
    
        
        
    
    
def _sparse_cca(S, c1=None, c2=None, n_iters=50, tol=1e-9):
    if c1 is None:
       c1 = np.sqrt(1 + np.sqrt(S.shape[1]))
    if c2 is None:
        c2 = np.sqrt(1 + np.sqrt(S.shape[1]))

    w2 = np.random.normal(0, 1, size=S.shape[1])
    w2/=np.linalg.norm(w2)
    objfunc_vals = np.zeros((n_iters))
    prev_f = -1e16
    for i in range(n_iters):
        w1_new, w2_new = weight_update(S, w2, c1, c2)
        curr_f = pobjfunc(w1_new, w2_new, S, c1, c2)
        if ((np.abs(curr_f - prev_f)<tol)|(curr_f<prev_f)):
            break 
        prev_f = curr_f
        w1, w2 = w1_new, w2_new
        objfunc_vals[i] = curr_f
    return w1, w2, objfunc_vals[:i]


def sparse_cca(S, c1=None, c2=None, n_starts=10, n_iters=50, tol=1e-9):
    starts = {'w1':[], 'w2':[], 'fmax':[], 'nit':[], 'r':[], 'p1':[], 'p2':[]}
    for i in range(n_starts):
        w1, w2, fvals = _sparse_cca(S, c1, c2, n_iters, tol)
        starts['w1'].append(w1)
        starts['w2'].append(w2)
        starts['fmax'].append(fvals[-1])
        starts['nit'].append(len(fvals))
        starts['r'].append(w1.T.dot(S).dot(w2))
        starts['p1'].append(np.sum(np.abs(w1)))
        starts['p2'].append(np.sum(np.abs(w2)))  
    return starts






class PCCA:
    
    def __init__(self, X=None, Y=None, S=None, ncomps=1):
        if (X is not None) and (Y is not None) and (S is None):
            S = base_utils.corr(X, Y)
        self.X, self.Y, self.ncomps = X, Y, ncomps
        self.S, self.cols1, self.cols2, self.is_pd = base_utils.check_type(S) 
        
    def _fit_comp(self, S, c1=None, c2=None, n_starts=1, n_iters=50, tol=1e-9,
             return_full=False):
        starts_dict = sparse_cca(S, c1, c2, n_starts,  n_iters, tol)
        starts_dfram = pd.DataFrame([starts_dict[x] for x in 
                                     ['fmax', 'nit', 'r', 'p1', 'p2']]).T

        wx = starts_dict['w1'][starts_dfram[0].idxmax()]
        wy = starts_dict['w2'][starts_dfram[0].idxmax()]
        
        if return_full:
            return wx, wy, starts_dfram
        else:
            return wx, wy, starts_dfram[0].max()
     
    def _fit(self, S, c1=None, c2=None, n_starts=1, n_iters=50, tol=1e-9):
        S0 = S.copy()
        Wx = np.zeros((S.shape[0], self.ncomps))
        Wy = np.zeros((S.shape[1], self.ncomps))
        
        for i in range(self.ncomps):
            wx, wy, f= self._fit_comp(S, c1, c2, n_starts, n_iters, tol)
            Wx[:, i] = wx
            Wy[:, i] = wy
            S = S - (wx[: ,None].T.dot(S0).dot(wy[:, None])) * (np.outer(wx, wy))
        return Wx, Wy
            
    def fit(self, c1=None, c2=None, n_starts=1, n_iters=50, tol=1e-9):
        Wx, Wy = self._fit(self.S, c1, c2, n_starts, n_iters, tol)
        self.Wx = Wx
        self.Wy = Wy
        
        if self.X is not None:
            self.Zx = self.X.dot(Wx)
        else:
            self.Zx = None

        if self.Y is not None:
            self.Zy = self.Y.dot(Wy)
        else:
            self.Zy = None
            
        if self.is_pd:
            self.Wx = pd.DataFrame(self.Wx, index=self.cols1)
            self.Wy = pd.DataFrame(self.Wy, index=self.cols2)
        if (self.X is not None) and (self.Y is not None):
            #Sx, Sy = base_utils.corr(self.X), base_utils.corr(self.Y)
            #vx = np.diag(np.sqrt(1.0 / np.diag(Wx.T.dot(Sx).dot(Wx))))
            #vy = np.diag(np.sqrt(1.0 / np.diag(Wy.T.dot(Sy).dot(Wy))))
            #Sxy = vx.dot(Wx.T.dot(self.S).dot(Wy)).dot(vy)
            Sxy = base_utils.corr(self.Zx, self.Zy)
        else:
            Sxy = None
        self.Sxy = Sxy
        
        
    def _cv_eval(self, Sxy):
        k = np.trace(Sxy)-np.trace(np.abs(np.flip(Sxy, axis=1)))
        return k
    
    def _cvfit(self, S, Xt, Yt, c1, c2, n_starts=1, n_iters=100, tol=1e-9):
        Wx, Wy = self._fit(S, c1, c2, n_starts, n_iters, tol)
        Zx, Zy = Xt.dot(Wx), Yt.dot(Wy)
        Sxy = base_utils.corr(Zx, Zy)
        return self._cv_eval(Sxy)
    
    def cross_validate(self, c1_vals=None, c2_vals=None, k_splits=7, vocal=True):
        
        if c1_vals is None:
            c1_vals = np.arange(0.2, 5)
            
        if c2_vals is None:
            c2_vals = np.arange(0.2, 5)
        
        c1grid, c2grid = np.meshgrid(c1_vals, c2_vals)            
        n = self.X.shape[0]
        n_per = np.round(n / k_splits)
        tmp = np.arange(0, n, n_per)
        tmp = np.concatenate([tmp, np.atleast_1d(n)])
        bounds = [(int(tmp[i]), int(tmp[i+1])) for i in range(k_splits)]
        indices = np.zeros((n,)).astype(bool)
        xval_dict = {'Xts':[], 'Yts':[], 'Ss':[]}
        for i in range(k_splits):
            a, b = bounds[i]
            indices[a:b] = True
            Xf, Yf = self.X[~indices], self.Y[~indices]
            Xt, Yt = self.X[indices], self.Y[indices]
            S = base_utils.corr(Xf, Yf)
            xval_dict['Xts'].append(Xt)
            xval_dict['Yts'].append(Yt)
            xval_dict['Ss'].append(S)
            indices[a:b] = False
        cvres = []
        cols = ['c1', 'c2']+['r%i'%i for i in range(1, k_splits+1)]
        grid = list(zip(c1grid.flatten(), c2grid.flatten()))
        count, ngr, durations = 0, len(grid), []
        sub_count = 0
        for c1i, c2i in grid:
            start = time.time()
            cvresi = [c1i, c2i]
            for i in range(k_splits):
                r = self._cvfit(xval_dict['Ss'][i],
                                xval_dict['Xts'][i], 
                                xval_dict['Yts'][i], c1i, c2i)
                cvresi.append(r)
            
            cvres.append(cvresi)
            end = time.time()
            duration = end - start
            durations.append(duration)
            count+=1
            sub_count+=1
            if sub_count==100:
                sub_count = 0
                if vocal:
                    print("%i/%i - %4.3f"%(count, ngr, np.mean(durations)*(ngr-count)))
        
        cvres = pd.DataFrame(cvres, columns=cols)
        cvres['mean'] = cvres.iloc[:, 2:].mean(axis=1)
        cvres['sd'] = cvres.iloc[:, 2:].std(axis=1)
        idxmax = cvres['mean'].idxmax()
        c1max, c2max = cvres.loc[idxmax, 'c1'], cvres.loc[idxmax, 'c2']
        self.c1max, self.c2max = c1max, c2max
        self.cvres = cvres
        self.durations = durations
        
    def permutation_test(self, c1, c2, n_samples=1000, vocal=True, n_starts=1,
                         n_iters=100, tol=1e-9):
        Xc, Yc = self.X.copy(), self.Y.copy()
        nc = self.ncomps
        k = int(nc * (nc + 1) / 2)
        rho_samples = np.zeros((n_samples, k))
        wx_samples =  np.zeros((n_samples, nc*Xc.shape[1]))
        wy_samples =  np.zeros((n_samples, nc*Yc.shape[1]))

        for i in range(n_samples):
            np.random.shuffle(Xc)
            np.random.shuffle(Yc)
            S = base_utils.corr(Xc, Yc)
            Wx, Wy = self._fit(S, c1, c2, n_starts, n_iters, tol)
            Zx, Zy = Xc .dot(Wx), Yc.dot(Wy)
            Sxy = base_utils.corr(Zx, Zy)
            rho_samples[i] = linalg_utils.vech(Sxy)
            wx_samples[i] = linalg_utils.vec(Wx)
            wy_samples[i] = linalg_utils.vec(Wy)
            if vocal:
                print(i)
        self.rho_samples = rho_samples
        self.wx_samples = wx_samples
        self.wy_samples = wy_samples
        
        
        
    
    
            

"""   


S = np.array([[1.0, 0.0, 0.9, 0.0],
              [0.0, 1.0, 0.0, 0.9],
              [0.9, 0.0, 1.0, 0.0],
              [0.0, 0.9, 0.0, 1.0]])
    
w = np.zeros((100, 2)) 
v = np.zeros((100, 2)) 

w[:10, 0] = 1.0
w[10:20, 0] = -1.0

w[80:90, 1] = 1.0
w[90:, 1] = -1.0

v[20:40, 0] = 1.0
v[85:90, 1] = -1.0

U = mv.multi_rand(S)



X = sp.stats.matrix_normal(U[:, :2].dot(w.T), np.eye(U.shape[0]),
                            np.eye(w.shape[0])).rvs()



Y = sp.stats.matrix_normal(U[:, 2:].dot(v.T), np.eye(U.shape[0]),
                            np.eye(w.shape[0])).rvs()

X, Y = mv.csd(X), mv.csd(Y)

pcca = PCCA(X, Y, ncomps=2)
pcca.fit(4.5, 6.5)
print(pcca.Sxy)
Wx, Wy = pcca.Wx, pcca.Wy
Sxy = pcca.Sxy
p, q = Sxy.shape
n, p = pcca.X.shape
statfunc_utils.multivariate_association_tests(np.diag(Sxy)[:1], p, q, n-p)
statfunc_utils.multivariate_association_tests(np.diag(Sxy)[1:], p, q, n-p)
statfunc_utils.multivariate_association_tests(np.diag(Sxy)[:, None].T, p, q, n-p)

pcca = PCCA(X, Y, ncomps=2)
pcca.cross_validate(np.linspace(6.0, 12.0, 150), np.linspace(4.0, 10.0, 150),
                    5, vocal=True)
cvres = pcca.cvres

pcca = PCCA(X, Y, ncomps=2)
pcca.permutation_test(8.2, 5.9)
rho_samples = pd.DataFrame(pcca.rho_samples)
wx_samples = pd.DataFrame(pcca.wx_samples)
wy_samples = pd.DataFrame(pcca.wy_samples)
pcca.fit(8.2, 5.9)

wx, wy = mv.vecc(pcca.Wx), mv.vecc(pcca.Wy)

res = np.block([[wx, wx_samples.std().values[:, None]],
                [wy, wy_samples.std().values[:, None]]])

res = pd.DataFrame(res, columns=['w', 'SE'])
res['t'] = res['w']/res['SE']



Xv, Yv = np.meshgrid(np.linspace(0.1, 9.0, 200), np.linspace(0.1, 9.0, 200))
res = []
for c1, c2 in list(zip(Xv.flatten(), Yv.flatten())):
    pcca.fit(c1, c2, n_starts=1)
    Sxy = pcca.Sxy
    k = np.trace(Sxy)-np.trace(np.abs(np.flip(Sxy, axis=1)))
    res.append([c1, c2, k])
    print("%3.3f-%3.3f-%3.3f"%(c1, c2, k))

resdf = pd.DataFrame(np.array(res))
c1_max, c2_max = resdf[0].loc[resdf[2].idxmax()], resdf[1].loc[resdf[2].idxmax()]
pcca.fit(c1_max, c2_max)

def func(params):
    pcca = PCCA(X, Y, ncomps=2)
    c1, c2 = params
    pcca.fit(c1, c2, n_starts=1)
    Sxy = pcca.Sxy
    k = np.trace(Sxy)
    return k

def cst(params):
    pcca = PCCA(X, Y, ncomps=2)
    c1, c2 = params
    pcca.fit(c1, c2, n_starts=1)
    Sxy = pcca.Sxy
    return  0.05 - np.trace(np.abs(np.flip(Sxy, axis=1)))


    
constr = [{'type':'ineq', 'fun':cst}]
    
optimzer = sp.optimize.minimize(func, [0.5, 0.5], constraints=constr,
                           bounds=[(0.01, 10), (0.01, 10)], method='slsqp')















w1, w2, fvals = _sparse_cca(X1, X2)
    
starts = sparse_cca(X1, X2, c1=4.5, c2=4.5, n_starts=250, n_iters=500, tol=1e-16)
    
start_res = pd.DataFrame([starts[x] for x in ['fmax', 'nit', 'r', 'p1', 'p2']]).T

w1 = starts['w1'][start_res[0].idxmax()]
w2 = starts['w2'][start_res[0].idxmax()]
 

"""





