#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:29:28 2019

@author: lukepinkel
"""
import pandas as pd
import numpy as np
from numpy import eye, kron
from mvpy.api import (vine_corr, multi_rand, center,
                            vech, jmat)
from scipy.linalg import block_diag
from scipy.optimize import minimize
from mvpy.api import LMM

import matplotlib.pyplot as plt
import seaborn as sns

def initialize_lmm(n_units=50, n_unit_obs=5, n_levels=2, n_level_effects=2):
    Sv = vine_corr(n_levels*n_level_effects, 2)
    Se = vine_corr(n_levels, 2)
    
    Wv = eye(n_units)
    We = eye(n_units*n_unit_obs)
    
    Vc = kron(Sv, Wv)
    Ve = kron(Se, We)
    
    Zi = np.concatenate([jmat(n_unit_obs), np.arange(n_unit_obs)[:, None]], axis=1)
    Z = block_diag(*[Zi for i in range(n_units*2)])
    beta = np.random.normal(size=(2, 1))
    X = np.concatenate([Zi for i in range(n_units*n_levels)])
    
    U = center(multi_rand(Vc))
    E = center(multi_rand(Ve, size=Ve.shape[1]*2))
    e = E[[0]].T
    u = U[[0]].T
    
    y = X.dot(beta)+Z.dot(u)+e
    x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])
    
    data = np.concatenate([y.reshape(n_units*n_unit_obs, 2, order='F'), x[:, None]], axis=1)
    data = pd.DataFrame(data, columns=['y1', 'y2', 'x1'])
    data['id'] = np.concatenate([jmat(n_unit_obs)*i for i in range(n_units)])
    fixed_effects = "~x1+1"
    random_effects = {"id":"~x1+1"}
    yvar = ['y1', 'y2']
    return fixed_effects, random_effects, yvar, data, Sv, Se
    
    
RC1, RC2 = [], []
gtols = []
i = 0
k = 0




while i<200:
    fixed_effects, random_effects, yvar, data, Sv, Se = initialize_lmm()
    model = LMM(fixed_effects, random_effects,  yvar, data)
    true_params = np.concatenate([vech(Sv), vech(Se)])
        
    model.fit(verbose=0)
    gtols.append([i, model.gnorm])
    if model.gnorm<1e-6:
        res = model.optimizer
        #res2 = model.optimizer
        #res = minimize(model.loglike, model.theta, bounds=model.bounds, 
        #           options={'verbose':0, 'maxiter':100}, method='trust-constr')
    
         
    
        rc1 = np.concatenate([res.x[:, None], true_params[:, None]], axis=1)
        #rc2 =  np.concatenate([res2.x[:, None], true_params[:, None]], axis=1)
        RC1.append(rc1)
        #RC2.append(rc2)
        i+=1
        print(i, "%4.E"%model.gnorm)
    else:
        k+=1
        print(k)
        continue


RC1_D = [x[:, 0] - x[:, 1] for x in RC1]
#RC2_D = [x[:, 0] - x[:, 1] for x in RC2]
RC1_D = np.concatenate([x[:, None] for x in RC1_D], axis=1).T
#RC2_D = np.concatenate([x[:, None] for x in RC2_D], axis=1).T
mfit = pd.DataFrame(gtols)
df1 = pd.DataFrame(RC1_D)
#df2 =  pd.DataFrame(RC2_D)
df1 = pd.DataFrame(df1.stack())
#df2 = pd.DataFrame(df2.stack())

df1['c'] = df1.index.get_level_values(1)
#df2['c'] = df2.index.get_level_values(1)
df1['method'] = [1]*len(df1)
#df2['method'] = [2]*len(df2)
#df = pd.concat([df1, df2], axis=0)

AE1 = np.concatenate([(x[:, 0] - x[:, 1])[:, None] for x in RC1], axis=1).T
#AE2 =  np.concatenate([(x[:, 0] - x[:, 1])[:, None] for x in RC2], axis=1).T
MAE1 = np.concatenate([abs(x[:, 0] - x[:, 1])[:, None] for x in RC1], axis=1).T
#MAE2 =  np.concatenate([abs(x[:, 0] - x[:, 1])[:, None] for x in RC2], axis=1).T
MSE1 = np.concatenate([(x[:, 0] - x[:, 1])[:, None]**2 for x in RC1], axis=1).T
#MSE2 =  np.concatenate([(x[:, 0] - x[:, 1])[:, None]**2 for x in RC2], axis=1).T
MRE1 = np.concatenate([((x[:, 0] - x[:, 1])/x[:, 1])[:, None] for x in RC1], axis=1).T
#MRE2 =  np.concatenate([((x[:, 0] - x[:, 1])/x[:, 1])[:, None] for x in RC2], axis=1).T



AE1 = pd.DataFrame(AE1) 
#AE2 = pd.DataFrame(AE2)
MAE1 = pd.DataFrame(MAE1)
#MAE2 = pd.DataFrame(MAE2)
MSE1 = pd.DataFrame(MSE1)
#MSE2 = pd.DataFrame(MSE2)
MRE1 = pd.DataFrame(MRE1)
#MRE2 = pd.DataFrame(MRE2)

AE1 = AE1.stack()
#AE2 = AE2.stack()
MAE1 = MAE1.stack()
#MAE2 = MAE2.stack()
MSE1 = MSE1.stack() 
#MSE2 = MSE2.stack()
MRE1 = MRE1.stack()
#MRE2 = MRE2.stack()


AE1 = pd.DataFrame(AE1)
#AE2=  pd.DataFrame(AE2)
MAE1 = pd.DataFrame(MAE1)
#MAE2 = pd.DataFrame(MAE2)
MSE1 = pd.DataFrame(MSE1)
#MSE2 = pd.DataFrame(MSE2)
MRE1 = pd.DataFrame(MRE1)
#MRE2 = pd.DataFrame(MRE2)



AE1['c'] = AE1.index.get_level_values(1)
#AE2['c'] = AE2.index.get_level_values(1)
MAE1['c'] = MAE1.index.get_level_values(1)
#MAE2['c'] = MAE2.index.get_level_values(1)
MSE1['c'] = MSE1.index.get_level_values(1)
#MSE2['c'] = MSE2.index.get_level_values(1)
MRE1['c'] = MRE1.index.get_level_values(1)
#MRE2['c'] = MRE2.index.get_level_values(1)

#AE1['method'], AE2['method'] = [1]*len(AE1), [2]*len(AE2)
#MAE1['method'], MAE2['method'] = [1]*len(MAE1), [2]*len(MAE2)
#MSE1['method'], MSE2['method'] = [1]*len(MAE1), [2]*len(MSE2)
#MRE1['method'], MRE2['method'] = [1]*len(MAE1), [2]*len(MRE2)

Acc = pd.concat([AE1, MAE1, MSE1, MRE1], axis=1)
Acc = Acc.iloc[:, [0, 1, 2, 4, 6]]
Acc.columns =[0, 'c', 1, 2, 3]
Acc.groupby(['c']).agg(['mean', 'median', 'std', 'skew', 'size'])
#AE = pd.concat([AE1, AE2], axis=0)
#MAE = pd.concat([MAE1, MAE2], axis=0)
#MSE = pd.concat([MSE1, MSE2], axis=0)
#MRE = pd.concat([MRE1, MRE2], axis=0)

#meanAE = AE.groupby(['c', 'method']).agg(['mean', 'median', 'std', 'skew', 'size'])
#meanMAE = MAE.groupby(['c', 'method']).agg(['mean', 'median', 'std', 'skew', 'size'])
#meanMSE = MSE.groupby(['c', 'method']).agg(['mean', 'median', 'std', 'skew', 'size'])
#meanMRE = MRE.groupby(['c', 'method']).agg(['mean', 'median', 'std', 'skew', 'size'])


fig, ax = plt.subplots()
sns.violinplot(x='c', y=0, data=df1)
#sns.violinplot(x='c', y=0, data=df2)
sns.set_style('darkgrid')


fig, ax = plt.subplots()
#sns.violinplot(x='c', y=0, hue='method', data=df, cut=0)

fig, ax = plt.subplots()
g = sns.pointplot(x='c', y=0, data=df1, join=False, estimator=np.median ,
capsize=0.1)
#g = sns.pointplot(x='c', y=0, hue='method', data=df, join=False,
#              dodge=0.1, capsize=.1, estimator=np.median)
g.axhline(0)

fig, ax = plt.subplots()
#sns.boxplot(x='c', y=0, hue='method', data=df)

#means = df.groupby(['c', 'method']).agg(['mean', 'median', 'std', 'skew', 'size'])



X = pd.DataFrame(np.concatenate([x[:, [1]].T for x in RC1], axis=0))
Y =  pd.DataFrame(np.concatenate([x[:, [0]].T for x in RC1], axis=0))

Z = X - Y


