#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:51:25 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
from mvpy.api import CLM
from scipy.stats import chi2 as chi2_dist

data = pd.read_csv("/users/lukepinkel/Downloads/wine.csv", index_col=0)
data['temp'] = data['temp'].replace({'cold':0, 'warm':1})
data['contact'] = data['contact'].replace({'no':0, 'yes':1})

clm_mod = CLM("rating ~ temp+contact-1", data=data)
clm_mod.fit()
clm_mod.res

clm_int  =CLM("rating ~ 1", data=data)
clm_int.fit()

LLN = clm_int.loglike(clm_int.params)
LLF = clm_mod.loglike(clm_mod.params)

LLR = LLN - LLF
LLRp = chi2_dist.sf(LLR, len(clm_mod.params)-1)