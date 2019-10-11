#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:33:56 2019

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from mvpy.api import FactorAnalysis, rotate
from mvpy.api import multi_rand, center
np.set_printoptions(suppress=True)

L = sp.stats.ortho_group.rvs(dim=10)[:, :4]
L = rotate(L, 'varimax')[0]
eigs = np.random.beta(a=1, b=5, size=(4,))
eigs = eigs[eigs.argsort()[::-1]]
eigs = eigs/np.sum(eigs)*4.0
Phi = sp.stats.random_correlation.rvs(eigs)
Phi = np.eye(4)
Psi = np.diag(np.random.uniform(low=0.1, high=.9, size=(10)))

S = L.dot(Phi).dot(L.T)+Psi

X = center(multi_rand(S))

model = FactorAnalysis(X, nfacs=4, orthogonal=True, unit_var=True)
model.fit(n_iters=5000)
Lambda_h = model.Lambda
Phi_h = model.Phi
Psi_h = model.Psi

np.linalg.norm(np.diag(Psi_h**2) - np.diag(Psi))
np.linalg.norm(Phi_h - Phi)
R = np.linalg.lstsq(Lambda_h, L)[0]
np.linalg.norm(Lambda_h.dot(R) - L)
np.linalg.norm(R.T.dot(R)- np.eye(4))
LR, R = rotate(Lambda_h, 'varimax')
LR_h = LR.dot(np.linalg.inv(np.array([[0, -1, 0, 0],
                 [0, 0, 0, -1],
                 [0, 0, -1, 0],
                 [-1, 0, 0, 0]])))
np.linalg.norm(LR_h-L)