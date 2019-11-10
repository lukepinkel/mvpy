#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 23:49:38 2019

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import mvpy.api as mv
from mvpy.examples.simulate_mixed_model import SimulatedMixedModel
 

def logit(u):
    return 1 / (1 + np.exp(-u))


n_isu, isu_obs = 200, 5
n_obs = n_isu * isu_obs
sglm = SimulatedMixedModel(n_obs)  
        
sglm.add_grouping(n_isu, isu_obs, False)      

sglm.data['time'] = np.concatenate([np.arange(x) for x in sglm.group_sizes['id1'].values()])
sglm.add_random_var({'id1':'~1'}, np.array([[1.0]]))
var = sp.stats.multivariate_normal(cov=mv.vine_corr(4, 3)).rvs(size=n_obs)

sglm.add_fixed_var(['x1', 'x2', 'x3'], var)
sglm.beta = np.random.normal(size=(3, 1))
sglm.add_y(0.5)
yr = sglm.data['y']
sglm.data['y'] = logit(mv.csd(sglm.data['y']))>np.random.uniform(0, 1, size=n_obs)
sglm.data['y']*=1

sglm.beta
glmm = mv.GLMM(sglm.fixed_effects+"-1", sglm.random_effects, "y", sglm.data,
                 mv.Bernoulli())

glmm.fit(n_iters=10)




data = pd.read_stata("https://stats.idre.ucla.edu/stat/data/hsbdemo.dta")
data.index = data.id
data['awards'] = data['awards'].apply(pd.to_numeric, errors='coerce')
data['cid']*=1.0

glmm_mod = mv.GLMM("~1+C(female)", {"cid":"~1"}, "awards", data, mv.Poisson())
glmm_mod.fit()
