#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:41:33 2019

@author: lukepinkel
"""

import numpy as np
import pandas as pd
import mvpy.api as mv

import importlib
importlib.reload(mv)
pd.set_option('display.expand_frame_repr', False)
vechS = [2.926, 1.390, 1.698, 1.628, 1.240, 0.592, 0.929,
         0.659, 4.257, 2.781, 2.437, 0.789, 1.890, 1.278, 0.949,
         4.536, 2.979, 0.903, 1.419, 1.900, 1.731, 5.605, 1.278, 1.004,
         1.000, 2.420, 3.208, 1.706, 1.567, 0.988, 3.994, 1.654, 1.170,
         3.583, 1.146, 3.649]

S = pd.DataFrame(mv.invech(np.array(vechS)), columns=['anti1', 'anti2',
                 'anti3', 'anti4', 'dep1', 'dep2', 'dep3', 'dep4'])
S.index = S.columns

X = mv.center(mv.multi_rand(S))
X += np.array([1.750, 1.928, 1.978, 2.322, 2.178, 2.489, 2.294, 2.222])

data = X[['anti1', 'anti2', 'anti3', 'anti4']]

Lambda = pd.DataFrame(np.eye(4), index=data.columns, columns=data.columns)
Beta = pd.DataFrame([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]], index=data.columns, columns=data.columns)
PH = Lambda.copy()
TH = Lambda.copy()*0.0

sem_model = mv.SEM(data, Lambda, Beta, PH=PH.values, TH=TH.values)
sem_model.fit()
data = X.copy()

