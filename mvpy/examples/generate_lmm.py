#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:45:42 2019

@author: lukepinkel
"""
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import mvpy.api as mv

def initialize_lmm(n_units=50, n_unit_obs=5, n_levels=2, n_level_effects=2, 
                   beta_params_v=2, beta_params_e=2, vscale=4, escale=2,
                   bscale=5):
    
    Sv = mv.vine_corr(n_levels*n_level_effects, beta_params_v)
    Se = mv.vine_corr(n_levels, beta_params_e)
    
    Dv = np.diag([vscale]*Sv.shape[0])
    Sv = Dv.dot(Sv).dot(Dv)
    
    De = np.diag([escale]*Se.shape[0])
    Se = De.dot(Se).dot(De)
    
    
    
    Wv = np.eye(n_units)
    We = np.eye(n_units*n_unit_obs)
    
    #Vc = np.kron(Sv, Wv)
    #Ve = np.kron(Se, We)
    
    Zi = np.concatenate([mv.jmat(n_unit_obs),
                         np.arange(n_unit_obs)[:, None]], axis=1)
    Z = sp.linalg.block_diag(*[Zi for i in range(n_units*n_levels)])
    beta = np.random.normal(size=(2, 1))*bscale
    X = np.concatenate([Zi for i in range(n_units*n_levels)])
    
    U = sp.stats.matrix_normal.rvs(np.zeros((Wv.shape[0], Sv.shape[0])), 
                                   Wv, Sv, size=1000)
                                   
    E = sp.stats.matrix_normal.rvs(np.zeros((We.shape[0], Se.shape[0])), 
                                   We, Se, size=1000)
    e = mv.vecc(E[0])
    u = mv.vecc(U[0])
    
    y = X.dot(beta)+Z.dot(u)+e
    x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])
    
    data = np.concatenate([y.reshape(n_units*n_unit_obs, n_levels, order='F'), x[:, None]], axis=1)
    data = pd.DataFrame(data, columns=["y%i"%i for i in range(1, 1+n_levels)]+['x1'])
    data['id'] = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
    fixed_effects = "~x1+1"
    random_effects = {"id":"~x1+1"}
    yvar = ["y%i"%i for i in range(1, 1+n_levels)]
    return fixed_effects, random_effects, yvar, data, Sv, Se