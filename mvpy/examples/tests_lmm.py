#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 19:23:30 2019

@author: lukepinkel
"""
import pandas as pd
import statsmodels.api as sma
import mvpy.api as mv
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
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

Parameter_Estimates = []

for i in range(100):
    ix = np.random.choice(len(data), len(data), replace=True)
    lmm_mod = mv.LMM(fe, re, y, data.iloc[ix])
    lmm_mod.fit()
    Parameter_Estimates.append(lmm_mod.params)

PEs = pd.DataFrame(np.vstack(Parameter_Estimates))
pstack = pd.DataFrame(PEs.stack()).reset_index()
pstack.columns = ['iteration', 'parameter', 'value']
sns.boxplot(x='parameter', y='value', data=pstack)
sns.pointplot(x='parameter', y='value', data=pstack, join=False)
sns.violinplot(x='parameter', y='value', data=pstack)

PEs.mean(axis=0)
PEs.std(axis=0)
A = np.eye(72)
for x in data['Litter'].unique():
    tmp = np.isin(data.Pig.unique(), data[data['Litter']==x].loc[:, 'Pig'].unique())
    tmp = np.arange(len(tmp))[tmp]
    for a1 in tmp:
        for a2 in tmp:
            A[a1, a2] = 0.5
            
    
    
fe = "~Time+Feed"
re = {"Pig": "~1"}
y = "Weight"
acov = {'Pig': A}
lmm_mod = mv.LMM(fe, re, y, data, acov=acov)
lmm_mod.fit()


