#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:45:36 2019

@author: lukepinkel
"""

import pandas as pd
import numpy as np

import seaborn as sns
from mvpy.api import LMM
import matplotlib.pyplot as plt
data = pd.read_csv("/users/lukepinkel/Downloads/wls2.csv", index_col=0)
data = data.drop('idpub.1', axis=1)

olsA = LM("IQ~1+C(Religiosity)*C(sexrsp)", data=data)
olsA.sumstats
olsA.beta


olsB = LM("attract~1+C(Religiosity)*C(sexrsp)", data=data)
olsB.sumstats
olsB.beta



olsC = LM("attract~1+C(Extraversion)*C(sexrsp)+IQ", data=data)
olsC.sumstats
olsC.beta
X = olsC.X