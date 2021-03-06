#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 02:26:18 2019

@author: lukepinkel
"""


import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import mvpy.api as mv
from ..utils.plotting_utils import lmplot
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(0)
n_groups = 5
n_group_obs = 500
n_obs = n_groups * n_group_obs
X = np.arange(-5, 5, 10.0/n_obs)[:, None]+np.random.normal(size=(n_obs, 1))
Z = np.kron(np.eye(n_groups), np.ones((n_group_obs, 1)))
u = np.arange(10, -10, -20.0/n_groups)[:, None]
b = np.ones((1, 1))

mu = X.dot(b)
y1 = sp.stats.matrix_normal(mu, np.eye(n_obs), np.eye(1)*2).rvs()
y2 = y1 + 0.59*Z.dot(u)

df = pd.DataFrame(np.hstack([X, Z.dot(np.arange(n_groups))[:, None], y2]),
                  columns=['x', 'id', 'y'])
df[['x', 'y']] = mv.csd(df[['x', 'y']])


fig, ax = plt.subplots(ncols=2)
lmplot('x', 'y', data=df, figax=(fig, ax[0]), )
lmplot('x', 'y', hue='id', data=df, figax=(fig, ax[1]))

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
fig.tight_layout()

df['y2'] = ((df['y']+np.random.normal(size=len(df)))/np.sqrt(2))
df['y3'] = (df['y2']>0)*1.0

sm.formula.logit('y3~x', data=df).fit().summary()
sm.formula.logit('y3~x+C(id)', data=df).fit().summary()
fam =mv.Bernoulli()
glmm = mv.GLMM("~x", {"id":"~1"}, "y3", data=df, fam=fam)
glmm.fit(n_iters=10)

print(glmm.res)
print(glmm.u)
print(glmm.b)

nu_hat = pd.DataFrame(glmm.predict(glmm.mod.X, glmm.mod.Z), columns=['y'])
yhat = (nu_hat>0.0)*1.0
    
fig, ax = plt.subplots(ncols=2)
ax[0].scatter(nu_hat, df[['y2']], alpha=0.2)
bp1 = ax[1].boxplot(nu_hat[df['y3']==1].values[:, 0], positions=[1], 
        widths=0.8, patch_artist=True)
bp2 = ax[1].boxplot(nu_hat[df['y3']==0].values[:, 0], positions=[0], 
        widths=0.8, patch_artist=True)


c1 = (0.12156862745098039, 0.4666666666666667,  0.7058823529411765)
c2 = (1.0, 0.4980392156862745, 0.054901960784313725)

for bp, color in list(zip([bp1, bp2], [c1, c2])):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='k')
    
    for patch in bp['boxes']:
        patch.set(facecolor=color, alpha=.9)   
    

mng = plt.get_current_fig_manager()
mng.window.showMaximized()
fig.tight_layout()









