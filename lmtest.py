#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:30:03 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np

import seaborn as sns
import mvt.api as mva
import matplotlib.pyplot as plt
data = pd.read_csv("/users/lukepinkel/Downloads/wls2.csv", index_col=0)
data = data.drop('idpub.1', axis=1)

olsA = mva.LM("IQ~1+C(Religiosity)*C(sexrsp)", data=data)
olsA.sumstats
olsA.beta


olsB = mva.LM("attract~1+C(Religiosity)*C(sexrsp)", data=data)
olsB.sumstats
olsB.beta



olsC = mva.LM("attract~1+C(Extraversion)*C(sexrsp)+IQ", data=data)
olsC.sumstats
olsC.beta
X = olsC.X