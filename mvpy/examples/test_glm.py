#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:10:36 2019

@author: lukepinkel
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import mvpy.api as mv
from mvpy.models import glm2
#Bernoulli 1
spector_data = sm.datasets.spector.load_pandas()

spector_data.exog = sm.add_constant(spector_data.exog)
data = pd.concat([spector_data.exog, spector_data.endog], axis=1)

smglm = sm.formula.logit("GRADE~GPA+TUCE+PSI", data=data).fit()
smglm.summary()

mod = mv.GLM("GRADE~GPA+TUCE+PSI", data=data, fam=mv.Bernoulli())
mod.fit()
mod.res
smglm.params

#Bernoulli 2
smglm = sm.formula.probit("GRADE~GPA+TUCE+PSI", data=data).fit()
smglm.summary()
mod = mv.GLM("GRADE~GPA+TUCE+PSI", data=data, fam=mv.Bernoulli(link=mv.ProbitLink()))
mod.fit()
mod.res
smglm.params

#Poisson 1

data = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")
data.index = data['id']

smglm = sm.formula.glm("num_awards~(prog)+math", data, family=sm.families.Poisson()).fit()

mod = mv.GLM("num_awards~(prog)+math", data=data, fam=mv.Poisson())
mod.fit()
mod.res
smglm.params
smglm.summary()


#Poisson 2
data = sm.datasets.fair.load_pandas().data

smglm = sm.formula.glm('affairs ~ rate_marriage + age + yrs_married',
              data=data, family=sm.families.Poisson()).fit()

mod = mv.GLM("affairs ~ rate_marriage + age + yrs_married", data=data, 
               fam=mv.Poisson())
mod.fit()
mod.res
smglm.summary()

#Gaussian 2
nobs2 = 100
x = np.arange(nobs2)
np.random.seed(54321)
X = np.column_stack((x,x**2))
X = sm.add_constant(X, prepend=False)
lny = np.exp(-(.03*x + .0001*x**2 - 1.0)) + .1 * np.random.rand(nobs2)

smglm = sm.GLM(lny, X, family=sm.families.Gaussian(sm.families.links.log)).fit()
print(smglm.summary())

data = pd.DataFrame(np.concatenate([X, lny[:, None]], axis=1))
data.columns = ['x1', 'x2', 'c', 'y']
mod = mv.GLM("y ~ x1 + x2", data=data, 
               fam=glm2.Normal(0.0010218, link=glm2.LogLink()))
mod.fit()






