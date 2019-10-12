#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:23:30 2019

@author: lukepinkel
"""

import statsmodels.api as sma
import mvpy.api as mv
import numpy as np
import seaborn as sns

data = sma.datasets.get_rdataset('dietox', 'geepack').data.dropna()

fe = "~Time"
re = {"Pig": "~1"}
y = "Weight"

lmm_mod = mv.LMM(fe, re, y, data)
lmm_mod.fit()

lmm_mod.params
lmm_mod.b
lmm_mod.res
mlm_mod = sma.formula.mixedlm("Weight~Time", re_formula="~1", groups="Pig",
                              data=data).fit()

mlm_mod.summary()
np.linalg.norm(data['Weight'] - mlm_mod.predict())
np.linalg.norm(data['Weight'] - mlm_mod.predict())

fe = "~Time"
re = {"Pig": "~1+Time"}
y = "Weight"

lmm_mod = mv.LMM(fe, re, y, data)
lmm_mod.fit()

lmm_mod.params
np.linalg.norm(lmm_mod.y - lmm_mod.X.dot(lmm_mod.b))
np.linalg.norm(lmm_mod.y - lmm_mod.X.dot(lmm_mod.b) - lmm_mod.Z.dot(lmm_mod.u))


fe = "~Time+Feed"
re = {"Pig": "~1+Time", "Litter": "~1"}
y = "Weight"

lmm_mod = mv.LMM(fe, re, y, data)
lmm_mod.fit()

lmm_mod.params
np.linalg.norm(lmm_mod.y - lmm_mod.X.dot(lmm_mod.b))
np.linalg.norm(lmm_mod.y - lmm_mod.X.dot(lmm_mod.b) - lmm_mod.Z.dot(lmm_mod.u))

sns.jointplot(lmm_mod.y, lmm_mod.X.dot(lmm_mod.b))
sns.jointplot(lmm_mod.y, lmm_mod.X.dot(lmm_mod.b) + lmm_mod.Z.dot(lmm_mod.u))
