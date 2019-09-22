#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 19:10:36 2019

@author: lukepinkel
"""

import pandas as pd
import statsmodels.api as sm
from mvpy.models import glm2
reload(glm2)

#Bernoulli
spector_data = sm.datasets.spector.load_pandas()

spector_data.exog = sm.add_constant(spector_data.exog)
data = pd.concat([spector_data.exog, spector_data.endog], axis=1)

smglm = sm.formula.logit("GRADE~GPA+TUCE+PSI", data=data).fit()
smglm.summary()

mod = glm2.GLM("GRADE~GPA+TUCE+PSI", data=data, fam=glm2.Bernoulli())
mod.fit()
mod.res
smglm.params
#Poisson 1

data = pd.read_csv("/users/lukepinkel/Downloads/poisson_sim.csv", index_col=0)

smglm = sm.formula.glm("num_awards~C(prog)+math", data=data,
                       family=sm.families.Poisson()).fit()

mod = glm2.GLM("num_awards~C(prog)+math", data=data, fam=glm2.Poisson())
mod.fit()
mod.res
smglm.params
smglm.summary()


#Poisson 2
data = sm.datasets.fair.load_pandas().data

smglm = sm.formula.glm('affairs ~ rate_marriage + age + yrs_married',
              data=data, family=sm.families.Poisson()).fit()

mod = glm2.GLM("affairs ~ rate_marriage + age + yrs_married", 
               data=data, 
               fam=glm2.Poisson())
mod.fit(verbose=2)
mod.res
smglm.summary()




