#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 05:12:00 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from mvpy.models.nb2 import NegativeBinomial

data = sm.datasets.fair.load_pandas().data

smglm = sm.formula.negativebinomial('affairs ~ rate_marriage + age + yrs_married',
              data=data).fit()


model = NegativeBinomial('affairs ~ rate_marriage + age + yrs_married', data=data)     
model.fit()
  

smglm.params.values
model.params