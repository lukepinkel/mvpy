#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 18:51:25 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np
import mvpy.models.clm as clm
import importlib
importlib.reload(clm)


data = pd.read_csv("/users/lukepinkel/Downloads/wine.csv", index_col=0)
data['temp'] = data['temp'].replace({'cold':0, 'warm':1})
data['contact'] = data['contact'].replace({'no':0, 'yes':1})

clm_mod = clm.CLM("rating ~ temp+contact-1", data=data)
clm_mod.fit()
clm_mod.res

