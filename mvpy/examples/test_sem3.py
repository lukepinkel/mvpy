#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:36:44 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import mvpy.api as mv
import importlib
importlib.reload(mv)
pd.set_option('display.expand_frame_repr', False)

Phi = np.diag(np.random.uniform(low=0.5, high=2.0, size=(4)))
Beta = np.array([[0, 0.0, 0, 0],
                 [1, 0.0, 0, 0],
                 [0, -1., 0, 0],
                 [0, 0.0, 1, 0]])
Lambda = np.zeros((12, 4))
Lambda[0:3, 0] = 1
Lambda[3:6, 1] = 1
Lambda[6:9, 2] = 1
Lambda[9:12, 3] = 1
Theta = np.diag(np.random.uniform(low=0.5, high=2.0, size=(12)))

Lvars = mv.center(mv.multi_rand(Phi))
Lvars = Lvars.dot(np.linalg.inv(mv.mat_rconj(Beta)))
Xi = mv.center(mv.multi_rand(Theta))
Z = Lvars.dot(Lambda.T) + Xi
Z = pd.DataFrame(Z, columns=['x%i'%i for i in range(1, 13)])
Beta = pd.DataFrame(Beta, index=['v%i'%i for i in range(1, 5)])
Beta.columns = Beta.index
Lambda = pd.DataFrame(Lambda, index=Z.columns, columns=Beta.index)
Theta = pd.DataFrame(Theta, index=Z.columns, columns=Z.columns)
Phi = pd.DataFrame(Phi, index=Beta.index, columns=Beta.columns)
B = np.linalg.inv(mv.mat_rconj(Beta))
S = Lambda.dot(B).dot(Phi.values).dot(B.T).dot(Lambda.T.values)+Theta.values
S.columns = S.index
Beta.iloc[2, 1] = 1.0
#Transposition required to get from the generative to the hypothesis matrix
sem_mod = mv.SEM(Z, Lambda, Beta.T, Theta, Phi)
sem_mod.free[8:11] = 0.0
sem_mod.params
sem_mod.fit(method='ML')

sem_mod.res