#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:53:45 2020

@author: lukepinkel
"""

import mvpy.api as mv
import pandas as pd
import numpy as np

data = pd.read_csv("/users/lukepinkel/Downloads/bollen.csv", index_col=0)
data = data[['x1', 'x2', 'x3', 'y1', 'y2', 'y3', 'y4', 'y5',
             'y6', 'y7', 'y8', ]]
L = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1],
              [0, 0, 1]])
B = np.array([[False, False, False],
              [True,  False, False],
              [True,  True, False]])
LA = pd.DataFrame(L, index=data.columns, columns=['ind60', 'dem60', 'dem65'])
BE = pd.DataFrame(B, index=LA.columns, columns=LA.columns)
S = data.cov()
Zg = ZR = data
Lambda=LA!=0
Beta=BE!=0 
Lambda, Beta = pd.DataFrame(Lambda), pd.DataFrame(Beta)
Lambda.columns = ['ind60', 'dem60', 'dem65']
Lambda.index = Zg.columns
Beta.columns = Lambda.columns
Beta.index = Lambda.columns
Theta = pd.DataFrame(np.eye(Lambda.shape[0]),
                     index=Lambda.index, columns=Lambda.index)
Theta.loc['y1', 'y5'] = 0.05
Theta.loc['y2', 'y4'] = 0.05
Theta.loc['y2', 'y6'] = 0.05
Theta.loc['y3', 'y7'] = 0.05
Theta.loc['y4', 'y8'] = 0.05
Theta.loc['y6', 'y8'] = 0.05
Theta.loc['y5', 'y1'] = 0.05
Theta.loc['y4', 'y2'] = 0.05
Theta.loc['y6', 'y2'] = 0.05
Theta.loc['y7', 'y3'] = 0.05
Theta.loc['y8', 'y4'] = 0.05
Theta.loc['y8', 'y6'] = 0.05
  
mod = mv.MLSEM(Zg, Lambda, 
               Beta, Theta.values, 
               fit_func='ML', wmat='normal')
mod.fit()


comp = pd.DataFrame(np.hstack([mv.vechc(mod.hessian(mod.free)),
                               mv.vechc(mv.fprime(mod.gradient, mod.free))]))


H = mod.hessian(mod.free)
H = pd.DataFrame(H, index=mod.res.index, columns=mod.res.index)



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
sem_mod = mv.MLSEM(Z, Lambda, Beta.T, Theta, Phi)
sem_mod.fit()


