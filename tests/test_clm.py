#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:35:45 2019

@author: lukepinkel
"""


import numpy as np
import pandas as pd
from mvpy.api import CLM

np.set_printoptions(suppress=True)
data = pd.read_csv("/users/lukepinkel/Downloads/wls2.csv",
                   index_col=0).drop('idpub.1', axis=1)
data = data.apply(pd.to_numeric, errors='coerce')
Z = data[['sexrsp', 'IQ', 'Number_of_Children']].dropna()
Z['Number_of_Children'].replace({12:8, 11:8, 10:8, 9:8}, inplace=True)
X = Z[['sexrsp', 'IQ']]
Y = Z[['Number_of_Children']]
clm_mod = CLM(X.values, Y.values)
clm_mod.fit()