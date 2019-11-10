#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:43:56 2019

@author: lukepinkel
"""
import patsy # analysis:ignore
import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import mvpy.api as mv # analysis:ignore
import seaborn as sns # analysis:ignore
import statsmodels.api as sm # analysis:ignore
from mvpy.utils import data_utils, linalg_utils # analysis:ignore

import matplotlib.pyplot as plt # analysis:ignore

def logit(u):
    return 1 / (1 + np.exp(-u))

class SimulatedMixedModel:
    
    def __init__(self, n_obs):
        self.n_obs = n_obs
        self.data = pd.DataFrame(np.empty((n_obs, 1)))
        self.n_re = 0
        self.group_sizes = {}
        self.Zdict = {}
        self.Udict = {}
        self.re_info = {}
        self.fixed_vars = []
        self.random_effects = {}
        self.params_list = []
        self.params = np.empty((1))
    
    def add_grouping(self, n_isu, isu_obs, random=False, id_name=None):
        if random == True:
            isu_size =  np.ones(n_isu)*isu_obs
            for i in range(5):
                ix = np.random.choice(n_isu, int(n_isu/2), False)
                isu_size[ix]  +=1
                isu_size[~ix] -=1
            id_ = np.arange(n_isu)
            id_ = [id_[i]*np.ones(int(isu_size[i])) for i in range(n_isu)]
            id_ = np.concatenate(id_)
                
        else:
            id_ = np.kron(np.arange(n_isu), np.ones(isu_obs))[:, None]
        
        if id_name is None:
            id_name = "id%i"%(self.n_re + 1)
        self.data[id_name] = id_
        gs = dict(zip(self.data.groupby(id_name).agg('size').index, 
                      self.data.groupby(id_name).agg('size').values))
        self.group_sizes[id_name] = gs
        self.re_info[id_name] = {'n_isu':n_isu, 'isu_obs':isu_obs}
    
    def add_random_var(self, re_dict, cov):
        re_name = list(re_dict.values())[0]
        Zij = patsy.dmatrix(re_name, data=self.data, return_type='dataframe')
        grouping =list(re_dict.keys())[0]
        Ji = data_utils.dummy_encode(self.data[grouping], complete=True)
        Zi = linalg_utils.khatri_rao(Ji.T, Zij.T).T
        self.Zdict[grouping] = Zi
        self.Z = np.concatenate(list(self.Zdict.values()), axis=1)
        self.re_info[grouping]['n_vars'] = Zij.shape[1]
        self.re_info[grouping]['cov'] = cov
        k = self.re_info[grouping]['n_isu']
        U = sp.stats.matrix_normal(colcov=cov, rowcov=np.eye(k)).rvs()
        self.Udict[grouping] = U
        self.random_effects[list(re_dict.keys())[0]] = list(re_dict.values())[0]

    
    def add_fixed_var(self, var_names, var):
        if type(var_names) is str:
            var_names = list(var_names)
        var = linalg_utils._check_2d(linalg_utils._check_np(var))
        for i in range(len(var_names)):
            self.data[var_names[i]] = var[:, i]
        self.fixed_vars += var_names
        self.fixed_effects = "~" + "+".join(var_names)
    
    def add_y(self, error_var=0.5):
        E = sp.stats.matrix_normal(rowcov=np.eye(self.n_obs)*error_var).rvs()
        U = mv.vecc(np.hstack(self.Udict.values()))
        Z = np.hstack(self.Zdict.values())
        Xb = self.data[self.fixed_vars].dot(self.beta)
        y = Xb + Z.dot(U) + E
        self.data['y'] = y
        self.params = [mv.vech(self.re_info[x]['cov']) 
                       for x in self.re_info.keys()]
        self.params_list.append(np.ones(1)*error_var)
        self.params_list.append(linalg_utils._check_1d(self.beta))
        self.params = np.concatenate(self.params_list)
 
        

        