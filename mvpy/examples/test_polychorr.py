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
import mvpy.models.lvcorr as mvl
import mvpy.utils.statfunc_utils as statfunc_utils

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import importlib
importlib.reload(mv)

data = sm.datasets.get_rdataset("bfi", package="psych")
df = data['data']

X = df.iloc[:, :-3].dropna()
Xcorr = X.corr()

S = mvl.mixed_corr(X)
            
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

statfunc_utils.srmr(factor_model.Sigma, factor_model.S.values, factor_model.df)
statfunc_utils.agfi(factor_model.Sigma, factor_model.S.values, factor_model.df)
factor_model.RMSEA
factor_model.chi2

LR = pd.DataFrame(mvf.rotate(factor_model.Lambda, 'varimax')[0], index=X.columns)
g = sns.clustermap(LR, vmin=-1, vmax=1, center=0.0, cmap=plt.cm.bwr, method='ward')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

LR = pd.DataFrame(mvf.rotate(factor_model.Lambda, 'equamax')[0], index=X.columns)
g = sns.clustermap(LR, vmin=-1, vmax=1, center=0.0, cmap=plt.cm.bwr, method='ward')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

LR = pd.DataFrame(mvf.rotate(factor_model.Lambda, 'quartimax')[0], index=X.columns)
g = sns.clustermap(LR, vmin=-1, vmax=1, center=0.0, cmap=plt.cm.bwr, method='ward')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)

LR = pd.DataFrame(mvf.rotate(factor_model.Lambda, 'promax')[0], index=X.columns)
g = sns.clustermap(LR, vmin=-1, vmax=1, center=0.0, cmap=plt.cm.bwr, method='ward')
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)





