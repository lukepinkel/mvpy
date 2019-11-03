#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:21:24 2019

@author: lukepinkel
"""
import numpy as np #analysis:ignore
import pandas as pd #analysis:ignore
import scipy as sp #analysis:ignore
import scipy.stats #analysis:ignore
import seaborn as sns #analysis:ignore
import mvpy.api as mv #analysis:ignore
import matplotlib.pyplot as plt #analysis:ignore
sns.set_style('darkgrid')



res = []
k = 0

for h2 in np.linspace(0.1, 0.9, 200):
    h = h2
    for i in range(10):
        Smz = np.eye(2) + np.flip(np.eye(2), axis=1)*0.5*h
        n_isu = 100
        Y_mz = sp.stats.matrix_normal(rowcov=np.ones(n_isu), colcov=Smz).rvs()
        
        Sdz = np.eye(2) + np.flip(np.eye(2), axis=1)*0.25*h
        Y_dz = sp.stats.matrix_normal(rowcov=np.ones(n_isu), colcov=Sdz).rvs()
        
        
        Y = np.concatenate([mv.vec(Y_mz.T), mv.vec(Y_dz.T)])
        Y += 2
        df = pd.DataFrame(Y, columns=['y'])
        df['id'] = np.kron(np.arange(200), np.ones(2))
        df['k'] = np.concatenate([np.ones(200)*0.5, np.ones(200)*0.25])
        
        fe = "~1"
        re = {"id":"k-1"}
        yv = "y"
        
        lmm = mv.LMM(fe, re, yv, data=df)
        lmm.fit(hess_opt=True, maxiter=500, verbose=0)
        vg = lmm.res.iloc[0, 0]
        ve = lmm.res.iloc[1, 0]
        vp = vg + ve
        h_est = vg / vp
        res.append([lmm.gnorm, vg, lmm.res.iloc[0, 1], 
                    ve, lmm.res.iloc[1,  1], vp, h, h_est])
        k+=1
        print(k)
        


data = pd.DataFrame(res, columns=['gnorm', 'Vg', 'SE(vg)', 'Ve', 
                                  'SE(ve)', 'Vp', 'theta', 'theta_hat'])
    
sns.jointplot('theta', 'theta_hat', data=data, stat_func=sp.stats.pearsonr)
sns.regplot('theta', 'theta_hat', data=data)

sns.regplot(np.percentile(data['theta'], np.linspace(0, 100, 1000)), 
           np.percentile(data['theta_hat'], np.linspace(0, 100, 1000)))


data['q'] = pd.qcut(data['theta'], 5)
sns.boxplot('q', 'theta_hat', data=data)

sns.residplot('theta', 'theta_hat', data)


