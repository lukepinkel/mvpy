#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:46:54 2019

@author: lukepinkel
"""
import numpy as np
import pandas as pd
from mvt.models import pls, sem, clm, factor_analysis, lmm, lvcorr, mv_rand
from mvt.models.mv_rand import vine_corr, multi_rand
from mvt.utils.base_utils import csd
from mvt.models.pls import PLS_SEM
from numpy.linalg import LinAlgError, norm
import seaborn as sns
import matplotlib.pyplot as plt

n=5000
v1 = np.random.rand(3, 1)
v2 = np.random.rand(3, 1)
v3 = np.random.rand(3, 1)
v4 = np.random.rand(3, 1)
v5 = np.random.rand(3, 1)
v6 = np.random.rand(3, 1)
v7 = np.random.rand(3, 1)
v8 = np.random.rand(3, 1)

S1 = vine_corr(3, 1)
S2 = vine_corr(3, 1)
S3 = vine_corr(3, 1)

X1 = csd(multi_rand(S1, size=n))
X2 = csd(multi_rand(S2, size=n))
X3 = csd(multi_rand(S3, size=n))


L1 = X1.dot(v1)
L2 = X2.dot(v2)
L3 = X2.dot(v3)

L4 = csd(1.0*L2+1.0*L3+np.random.normal(size=(n, 1)))
L5 = csd(1.0*L1+1.0*L2+1.0*L4+np.random.normal(size=(n, 1)))

L6 = csd(1.0*L1+1.0*L5+np.random.normal(size=(n, 1)))
L7 = csd(1.0*L4+1.0*L3+np.random.normal(size=(n, 1)))
L8 = csd(1.0*L5+1.0*L4+1.0*L7+1.0*L6+np.random.normal(size=(n, 1)))


X3 = csd(L3.dot(v3.T)+np.random.normal(size=(n, 3)))
X4 = csd(L4.dot(v4.T)+np.random.normal(size=(n, 3)))
X5 = csd(L5.dot(v5.T)+np.random.normal(size=(n, 3)))
X6 = csd(L6.dot(v6.T)+np.random.normal(size=(n, 3)))
X7 = csd(L7.dot(v7.T)+np.random.normal(size=(n, 3)))
X8 = csd(L8.dot(v8.T)+np.random.normal(size=(n, 3)))

X1 = pd.DataFrame(X1, columns=['x01', 'x02', 'x03'])
X2 = pd.DataFrame(X2, columns=['x04', 'x05', 'x06'])
X3 = pd.DataFrame(X3, columns=['x07', 'x08', 'x09'])
X4 = pd.DataFrame(X4, columns=['x10', 'x11', 'x12'])
X5 = pd.DataFrame(X5, columns=['x13', 'x14', 'x15'])
X6 = pd.DataFrame(X6, columns=['x13', 'x14', 'x15'])
X7 = pd.DataFrame(X7, columns=['x13', 'x14', 'x15'])
X8 = pd.DataFrame(X8, columns=['x13', 'x14', 'x15'])

X_blocks = [X1, X2, X3, X4, X5, X6, X7, X8]
predictor_matrix = np.array([[0, 0, 0, 0, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 1, 1],
                             [0, 0, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0, 0]])


modes = ['B', 'B', 'B', 'A', 'A', 'A', 'A', 'A']
wtrue = pd.DataFrame(np.concatenate([v1, v2, v3, v4, v5, v6, v7, v8]),
                     index=['x%s'%str(i).zfill(2) for i in range(1, 25)])
Rlv = pd.DataFrame(np.block([L1, L2, L3, L4, L5, L6, L7, L8])).corr()

n_boot = 50
pls_mod = PLS_SEM(X_blocks, predictor_matrix, modes)
pls_mod.fit(boot_vocal=True, n_boot=n_boot)
print(pls_mod.Beta)
W_samples, B_samples = pls_mod.W_samples, pls_mod.Beta_samples


B_samples = pd.DataFrame(B_samples.reshape(64, n_boot).T)
B_samples = B_samples.loc[:, B_samples.sum(axis=0)!=0]
W_samples = pd.DataFrame(W_samples.reshape(24, n_boot).T)

for x in B_samples.columns:
    sns.distplot(B_samples[[x]])
sns.distplot(B_samples[[4]], bins=200)

B_samples.plot.kde()
W_samples.plot.kde()


GoF = []
uix = np.triu_indices(8, 1)
pmat = predictor_matrix.copy()

for i in range(1000):
    rnd = np.random.binomial(1, .5, size=(28))
    while np.sum(rnd)<10:
        rnd = np.random.binomial(1, .5, size=(28))
    pmat[uix] = np.random.permutation(predictor_matrix.copy()[uix])
    pmat[uix] = rnd
    pls_mod3 = PLS_SEM(X_blocks, pmat, modes)
    try:
        pls_mod3.fit(n_boot=1)
        GoF.append([pls_mod3.GoF, norm(pmat-predictor_matrix)])
    except LinAlgError:
        continue
    print(i)

GoFdf = pd.DataFrame(GoF).dropna()

sns.kdeplot(GoFdf)
sns.jointplot(x=0, y=1, data=GoFdf)
sns.regplot(GoFdf[0], GoFdf[1])



GFR = GoFdf.groupby(1)[0].agg(['mean', 'std', 'size'])


data = GoFdf.copy()[(GoFdf[1]>2.6)&(GoFdf[1]<4.69)]
data.columns=['x', 'g']
data = data[['g', 'x']]

pal = sns.cubehelix_palette()
g = sns.FacetGrid(data, row="g", hue="g", aspect=15, size=0.6, palette=pal)
g.map(sns.kdeplot, "x", shade=True, alpha=1, lw=0.1, bw=.005,)
g.map(plt.axvline, x=0.12, lw=2.0, color='k')
g.set(xlim=(0.02, 0.2))
g.map(plt.axhline, y=0, lw=2, clip_on=False)
g.fig.subplots_adjust(hspace=-0.1)
g.set(title="")
g.set(yticks=[])

