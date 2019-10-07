#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:39:24 2019

@author: lukepinkel
"""
import numpy as np
import pandas as pd
import mvpy.api as mv
import mvpy.models.factor_analysis as mvf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
importlib.reload(mv)

data = sm.datasets.get_rdataset("bfi", package="psych")
df = data['data']

X = df.iloc[:, :-3].dropna()
Xcorr = X.corr()

R = np.zeros((25, 25))
k=0
for i in range(25):
    for j in range(i, 25):
        if i==j:
            R[i, j] = 0.5
        else:
            R[i, j] = mv.polychorr(X.iloc[:, i], X.iloc[:, j])[0]
        print("%i-%u; %i/%i; (%2.2f):(%2.2f) - %2.2f"%(i, j, k, 325,
              Xcorr.iloc[i, j], R[i, j],np.linalg.norm(Xcorr.iloc[i, j]-R[i, j])))
        k+=1


            
S = pd.DataFrame(R+R.T, index=X.columns, columns=X.columns)
sns.clustermap(S, vmin=-1, vmax=1, center=0, cmap=plt.cm.bwr, method='ward')

Xsim = mv.center(mv.multi_rand(S))

factor_model = mvf.FactorAnalysis(Xsim, 5)
factor_model.fit()
factor_model.gradient(factor_model.free)
factor_model.chi2
factor_model.stdchi2
factor_model.GFI
factor_model.RMSEA
Lambda = factor_model.Lambda
Lambda, _ = mvf.rotate(Lambda, 'varimax')
Lambda = pd.DataFrame(Lambda, index=X.columns)
sns.heatmap(Lambda, vmin=-1, vmax=1, center=0.0, cmap=plt.cm.seismic)