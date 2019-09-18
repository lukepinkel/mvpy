#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:33:56 2019

@author: lukepinkel
"""

import numpy as np
import scipy as sp
from mvpy.models.factor_analysis import FactorAnalysis
from mvpy.api import vine_corr, multi_rand, center
np.set_printoptions(suppress=True)

L = sp.stats.ortho_group.rvs(dim=10)[:, :4]
eigs = np.random.beta(a=1, b=5, size=(4,))
eigs = eigs[eigs.argsort()[::-1]]
eigs = eigs/np.sum(eigs)*4.0
Phi = sp.stats.random_correlation.rvs(eigs)
Psi = np.diag(np.random.uniform(low=0.1, high=.9, size=(10)))

S = L.dot(Phi).dot(L.T)+Psi

X = center(multi_rand(S))

model = FactorAnalysis(X, nfacs=4, orthogonal=False, unit_var=True)
model.fit(n_iters=500)
Lambda_h = model.Lambda
Phi_h = model.Phi
Psi_h = model.Psi

np.diag(Psi_h**2) - np.diag(Psi)
