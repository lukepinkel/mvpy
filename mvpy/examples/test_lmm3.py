#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:26:58 2019

@author: lukepinkel
"""



import pandas as pd
import mvpy.api as mv
import numpy as np
import scipy as sp
import scipy.stats
import seaborn as sns

'''
This example demonstrates how one would go about adding a unique 
fixed effect for each dependent variable, which is a little messy,
but nevertheless feasable; future updates to make this easy are planned
'''
K = np.array([[1.0, 1.0, 0.5, 0.5, 0.125],
              [1.0, 1.0, 0.5, 0.5, 0.125],
              [0.5, 0.5, 1.0, 0.5, 0.125],
              [0.5, 0.5, 0.5, 1.0, 0.125],
              [0.125, 0.125, 0.125, 0.125, 1.0]])

K = np.array([[1.0, 1.0, 0.5, 0.5, ],
              [1.0, 1.0, 0.5, 0.5,],
              [0.5, 0.5, 1.0, 0.5,],
              [0.5, 0.5, 0.5, 1.0]])
n = 100
n2 = n*K.shape[0]
A = np.kron(np.eye(n), K)
s = 5
e = 5
R = np.eye(n2)*e
G = s*A

Z = np.eye(n2)

u = sp.stats.multivariate_normal.rvs(np.zeros(n2), G)
e = sp.stats.multivariate_normal.rvs(np.zeros(n2), R)

y = Z.dot(u)+e+2

df = pd.DataFrame(y, columns=['y'])
df['id'] = np.arange(n2)
acov = {'id': A}

lmm_mod = mv.LMM("~1", {"id":"~1"}, yvar="y", data=df, acov=acov)
lmm_mod.fit(hess_opt=True)



K = np.array([[1.0, 1.0, 0.5, 0.5, ],
              [1.0, 1.0, 0.5, 0.5,],
              [0.5, 0.5, 1.0, 0.5,],
              [0.5, 0.5, 0.5, 1.0]])
n = 100
n2 = n*K.shape[0]
A = np.kron(np.eye(n), K)
s = np.diag([2, 2]).dot(mv.vine_corr(2, 5)).dot(np.diag([2, 2]))
e = np.diag([3, 1])
R = np.kron(e, np.eye(n2))
G = np.kron(s, A)

Z = np.eye(n2*2)

u = sp.stats.multivariate_normal.rvs(np.zeros(n2*2), G)
e = sp.stats.multivariate_normal.rvs(np.zeros(n2*2), R)
X = np.zeros((800, 2))
X[0:400, 0] = 1
X[400:, 1] = 1
y = Z.dot(u)+e+X.dot(np.array([-1, 1]))

df = pd.DataFrame(mv.invec(y, n2, 2), columns=['y1', 'y2'])
df['id'] = np.arange(n2)
acov = {'id': A}

lmm_mod = mv.LMM("~1", {"id":"~1"}, yvar=["y1", "y2"], data=df, acov=acov)

X, Z, y = lmm_mod.X, lmm_mod.Z, lmm_mod.y
X[400:] = -1
lmm_mod.X = X
lmm_mod.XZY = np.block([X, Z, y])
lmm_mod.XZ = np.block([X, Z])
lmm_mod.A = np.block([[X, Z], [np.zeros((Z.shape[1], X.shape[1])),
               np.eye(Z.shape[1])]])





lmm_mod.fit(hess_opt=True)

sns.jointplot(lmm_mod.y, lmm_mod.X.dot(lmm_mod.b) + lmm_mod.Z.dot(lmm_mod.u),
              stat_func=sp.stats.pearsonr)

sns.jointplot(lmm_mod.y, lmm_mod.X.dot(lmm_mod.b),
              stat_func=sp.stats.pearsonr)
