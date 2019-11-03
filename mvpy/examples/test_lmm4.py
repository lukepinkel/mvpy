#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:55:43 2019

@author: lukepinkel
"""
import numpy as np #analysis:ignore
import scipy as sp #analysis:ignore
import pandas as pd #analysis:ignore
import scipy.stats  #analysis:ignore
import seaborn as sns  #analysis:ignore

import statsmodels.api as sm #analysis:ignore
import mvpy.api as mv #analysis:ignore
from mvpy.examples import generate_lmm as glmm #analysis:ignore
import time#analysis:ignore
import matplotlib.pyplot as plt

def ttest(x, y):
    return sp.stats.ttest_ind(x, y)[0]

def mean_diff(x, y):
    return np.mean(x) - np.mean(y)


def median_diff(x, y):
    return np.median(x) - np.median(y)


def bootstrap_t(x, y, n_samples=5000, estimator=ttest):
    nx, ny = x.shape[0], y.shape[0]
    mu_x, mu_y = np.mean(x), np.mean(y)
    mu_p = (mu_x*nx+mu_y*ny)/(nx+ny)
    
    xp, yp = x - mu_x + mu_p, y - mu_y + mu_p
    
    t = []
    for i in range(n_samples):
        xix = np.random.choice(nx, nx, replace=True)
        yix = np.random.choice(ny, ny, replace=True)
        
        xpi, ypi = xp[xix], yp[yix]
        t.append(estimator(xpi, ypi))
    t_obs = estimator(x, y)
    t = np.array(t)
    p_boot = 1.0 - np.sum(t_obs>t)/n_samples
    return t_obs, p_boot, t

res = []

k = 1
mean_dur = 0.0
#6000 iterations -- 2 hours
for n_units in range(50, 100, 25): #2
    #n_units=50
        for n_unit_obs in range(4, 6):  #2
            #n_unit_obs=5
            n_levels=1
            n_level_effects=2
            for beta_params_v in range(2, 20, 3): #6
                #beta_params_v=2
                beta_params_e=2
                vscale=4
                escale=2
                bscale=2
                    
                
                for i in range(50): #10
                    start = time.time()
                    Sv = mv.vine_corr(n_levels*n_level_effects, beta_params_v)
                    Se = mv.vine_corr(n_levels, beta_params_e)
                    
                    Dv = np.diag([vscale]*Sv.shape[0])
                    Sv = Dv.dot(Sv).dot(Dv)
                    
                    De = np.diag([escale]*Se.shape[0])
                    Se = De.dot(Se).dot(De)
                    
                    
                    
                    Wv = np.eye(n_units)
                    We = np.eye(n_units*n_unit_obs)
                    
                    #Vc = np.kron(Sv, Wv)
                    #Ve = np.kron(Se, We)
                    
                    Zi = np.concatenate([mv.jmat(n_unit_obs),
                                         np.arange(n_unit_obs)[:, None]], axis=1)
                    Z = sp.linalg.block_diag(*[Zi for i in range(n_units*n_levels)])
                    beta = np.random.normal(size=(n_level_effects, 1))*bscale
                    X = np.concatenate([Zi for i in range(n_units*n_levels)])
                    
                    U = sp.stats.matrix_normal.rvs(np.zeros((Wv.shape[0], Sv.shape[0])), 
                                                   Wv, Sv, size=1000)
                                                   
                    E = sp.stats.matrix_normal.rvs(np.zeros((We.shape[0], Se.shape[0])), 
                                                   We, Se, size=1000)
                    e = mv.vecc(E[0])
                    u = mv.vecc(U[0])
                    
                    y = X.dot(beta)+Z.dot(u)+e
                    x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])
                    
                    data = np.concatenate([y.reshape(n_units*n_unit_obs, n_levels, order='F'), x[:, None]], axis=1)
                    data = pd.DataFrame(data, columns=["y%i"%i for i in range(1, 1+n_levels)]+['x1'])
                    data['id'] = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
                    fixed_effects = "~x1+1"
                    random_effects = {"id":"~x1+1"}
                    yvar = ["y%i"%i for i in range(1, 1+n_levels)]
                    true_params = np.concatenate([mv.vech(Sv), mv.vech(Se), beta[:, 0]])
                    
                    
                    lmm = mv.LMM(fixed_effects, random_effects, yvar, data)
                    lmm.fit(hess_opt=True, verbose=0)
                    
                    param_ests = np.concatenate([lmm.params, lmm.b[:, 0]])
                    
                    mlm = sm.MixedLM.from_formula("y1~x1+1", re_formula="~x1+1", groups="id",
                                                  data=data).fit()
                    mlm_est = np.flip(mlm.params.values[2:]*mlm.scale)
                    mlm_est = np.concatenate([mlm_est, np.array([mlm.scale]), mlm.fe_params])
                    
                    e1 = np.linalg.norm(mlm_est - true_params)
                    e2 = np.linalg.norm(param_ests - true_params)
                    res.append([n_units, n_unit_obs, beta_params_v, e1, e2])
                    end = time.time()
                    duration = end - start
                    mean_dur = (mean_dur * k + duration) / (k + 1)
                    k+=1
                    print(i, mean_dur, (6000-k)*mean_dur/60/60)

for beta_params_v in range(10, 40, 5): #6
                #beta_params_v=2
                beta_params_e=2
                vscale=4
                escale=2
                bscale=2
                for i in range(50): #50
                    start = time.time()
                    Sv = mv.vine_corr(n_levels*n_level_effects, beta_params_v)
                    Se = mv.vine_corr(n_levels, beta_params_e)
                    
                    Dv = np.diag([vscale]*Sv.shape[0])
                    Sv = Dv.dot(Sv).dot(Dv)
                    
                    De = np.diag([escale]*Se.shape[0])
                    Se = De.dot(Se).dot(De)
                    
                    
                    
                    Wv = np.eye(n_units)
                    We = np.eye(n_units*n_unit_obs)
                    
                    #Vc = np.kron(Sv, Wv)
                    #Ve = np.kron(Se, We)
                    
                    Zi = np.concatenate([mv.jmat(n_unit_obs),
                                         np.arange(n_unit_obs)[:, None]], axis=1)
                    Z = sp.linalg.block_diag(*[Zi for i in range(n_units*n_levels)])
                    beta = np.random.normal(size=(n_level_effects, 1))*bscale
                    X = np.concatenate([Zi for i in range(n_units*n_levels)])
                    
                    U = sp.stats.matrix_normal.rvs(np.zeros((Wv.shape[0], Sv.shape[0])), 
                                                   Wv, Sv, size=1000)
                                                   
                    E = sp.stats.matrix_normal.rvs(np.zeros((We.shape[0], Se.shape[0])), 
                                                   We, Se, size=1000)
                    e = mv.vecc(E[0])
                    u = mv.vecc(U[0])
                    
                    y = X.dot(beta)+Z.dot(u)+e
                    x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])
                    
                    data = np.concatenate([y.reshape(n_units*n_unit_obs, n_levels, order='F'), x[:, None]], axis=1)
                    data = pd.DataFrame(data, columns=["y%i"%i for i in range(1, 1+n_levels)]+['x1'])
                    data['id'] = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
                    fixed_effects = "~x1+1"
                    random_effects = {"id":"~x1+1"}
                    yvar = ["y%i"%i for i in range(1, 1+n_levels)]
                    true_params = np.concatenate([mv.vech(Sv), mv.vech(Se), beta[:, 0]])
                    
                    
                    lmm = mv.LMM(fixed_effects, random_effects, yvar, data)
                    lmm.fit(hess_opt=True, verbose=0)
                    
                    param_ests = np.concatenate([lmm.params, lmm.b[:, 0]])
                    
                    mlm = sm.MixedLM.from_formula("y1~x1+1", re_formula="~x1+1", groups="id",
                                                  data=data).fit()
                    mlm_est = np.flip(mlm.params.values[2:]*mlm.scale)
                    mlm_est = np.concatenate([mlm_est, np.array([mlm.scale]), mlm.fe_params])
                    
                    e1 = np.linalg.norm(mlm_est - true_params)
                    e2 = np.linalg.norm(param_ests - true_params)
                    res.append([n_units, n_unit_obs, beta_params_v, e1, e2])
                    end = time.time()
                    duration = end - start
                    mean_dur = (mean_dur * k + duration) / (k + 1)
                    k+=1
                    print(k, i, mean_dur, (6000-k)*mean_dur/60/60)
               
df = pd.DataFrame(res, columns=['isu', 'nso', 'betav', 'mlm_est', 'lmm_est'])

df[['mlm_est', 'lmm_est']].agg(['mean', 'median', 'std', 'skew', 'kurtosis'])
'''
           mlm_est   lmm_est
mean      7.138989  4.657016
median    6.611100  4.350974
std       3.370245  2.162221
skew      0.815784  0.972845
kurtosis  0.730076  1.480116
'''

sp.stats.ttest_ind(df['mlm_est'], df['lmm_est'])
sp.stats.ranksums(df['mlm_est'], df['lmm_est'])
sp.stats.wilcoxon(df['mlm_est'], df['lmm_est'])

'''
Ttest_indResult(statistic=33.60881410344904, pvalue=1.1467860782256504e-226)
RanksumsResult(statistic=30.611487023208614, pvalue=8.608320667007138e-206)
WilcoxonResult(statistic=647665.0, pvalue=2.9629748704095283e-237)
'''
dfs = pd.DataFrame(df[['mlm_est', 'lmm_est']].stack().reset_index())
dfs.columns =['t', 'method', 'error']


fig, ax = plt.subplots(ncols=3)


sns.boxplot(x='method', y='error', data=dfs, ax=ax[0])
sns.violinplot(x='method', y='error', data=dfs, ax=ax[1])
sns.pointplot(x='method', y='error', data=dfs, join=False, ax=ax[2])
sns.pointplot(x='method', y='error', data=dfs, join=False, estimator=np.median, ax=ax[2])

fig, ax = plt.subplots()

sns.distplot(dfs[dfs['method']=='mlm_est']['error'], ax=ax)
sns.distplot(dfs[dfs['method']=='lmm_est']['error'], ax=ax)





df['diff'] = df["mlm_est"] - df["lmm_est"]


ols = sm.formula.ols("diff~isu+nso+betav", data=df).fit()
ols.summary()
sns.distplot(df['diff'])
fig, ax = plt.subplots(ncols=4)
sns.boxplot("betav", "diff", data=df, ax=ax[0])
sns.pointplot("betav", "diff", data=df, ax=ax[1])


sns.pointplot("betav", "diff", data=df, ax=ax[2])
ax[2].axhline(0)
sns.swarmplot("betav", "diff", data=df, ax=ax[3])







