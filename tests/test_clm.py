#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:40:15 2019

@author: lukepinkel
"""
import numpy as np
import pandas as pd
from mvpy.api import CLM, LM

np.set_printoptions(suppress=True)

data = pd.read_csv("/users/lukepinkel/clm_example.csv",
                   index_col=0)


clm_mod = CLM("Religiosity ~ IQ + sexrsp-1", data=data)
clm_mod.fit(optim='single')
print(clm_mod.res)

lm1 = LM("Religiosity~IQ+C(sexrsp)", data=data)
print(lm1.res)
print(lm1.sumstats)


lm2 = LM("IQ~C(Religiosity)*C(sexrsp)", data=data)
print(lm2.res)
print(lm2.sumstats)

lm3 = LM("IQ~Religiosity+C(sexrsp)", data=data)
print(lm3.res)
print(lm3.sumstats)


