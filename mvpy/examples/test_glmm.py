#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 11:55:30 2020

@author: lukepinkel
"""

import numpy as np
import sicpy as sp
import pandas as pd
import mvpy.api as mv

np.random.seed(240)
clusters = 70
cluster_obs = 20
n_obs = int(cluster_obs*clusters)
df = pd.DataFrame(np.zeros((n_obs, 6)), columns=['const', 'x1', 'x2', 'x3','x4', 'z'])
df['const'] = 1
X = sp.stats.multivariate_normal(np.zeros(4), mv.vine_corr(4, 10)).rvs(n_obs)
df['x1'] = X[:, 0]
df['x2'] = X[:, 1]
df['x3'] = X[:, 2]
df['x4'] = X[:, 3]
df['x5'] = np.tile(np.kron(np.arange(2), np.ones(int(cluster_obs/2))), clusters)
df['z'] = np.kron(np.arange(clusters), np.ones(cluster_obs))

b = np.array([1, -1, 0.5, -0.5, 2])

df['y'] = np.zeros(n_obs)


model = mv.LMM("~x1+x2+x3+x3+x4", {"z":"~1+x3+x5"}, "y", data=df)
X = model.X
Z = model.Z
S = mv.invech(np.array([1.0, -0.3, 0.3, 1.0, 0.007, 1.0]))
v = np.diag([2, np.sqrt(2), 1.0])
S = v.dot(S).dot(v)
U = sp.stats.multivariate_normal(np.zeros(3), S).rvs(clusters)
u = mv.vec(U)


e = sp.stats.logistic().rvs(n_obs)
nu =Z.dot(u)+X.dot(b)
v = nu+e
y = (v>0)*1.0

df['y'] =y
df['v'] = v

glmm = mv.GLMM("~x1+x2+x3+x4", {"z":"~1+x3+x5"}, "y", data=df, fam=mv.Binomial())
glmm.fit()
    
   
