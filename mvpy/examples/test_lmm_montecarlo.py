#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:46:21 2019

@author: lukepinkel
"""
import numpy as np#analysis:ignore
import pandas as pd #analysis:ignore
import mvpy.examples.generate_lmm as mclmm #analysis:ignore
import mvpy.api as mv #analysis:ignore
import seaborn as sns #analysis:ignore
import matplotlib.pyplot as plt #analysis:ignore
from mvpy.utils import linalg_utils #analysis:ignore

np.set_printoptions(suppress=True)#analysis:ignore
sns.set_style('darkgrid')


estimate_error = np.zeros((1000, 14))
i=0
maximum = np.zeros(13)
minimum = np.zeros(13)

while i<1000:
    fe, re, yvar, data, Sv, Se = mclmm.initialize_lmm()

    lmm = mv.LMM(fe, re, yvar, data)
    lmm.fit(maxiter=300, verbose=0)
    error = lmm.params - np.concatenate([mv.vech(Sv), mv.vech(Se)])
    Verr, Serr = mv.invech(error[:-3]), mv.invech(error[-3:])
    dv, de = np.diag(np.sqrt(1.0/np.diag(Sv))), np.diag(np.sqrt(1.0/np.diag(Se)))
    Verr, Serr = dv.dot(Verr).dot(dv), de.dot(Serr).dot(de)
    error = np.concatenate([mv.vech(Verr), mv.vech(Serr)])
    estimate_error[i, :13] = error
    estimate_error[i, 13] = lmm.gnorm
    
    if lmm.gnorm<1e-3:
        maximum = np.maximum(maximum, np.concatenate([mv.vech(Sv), mv.vech(Se)]))
        minimum = np.minimum(minimum, np.concatenate([mv.vech(Sv), mv.vech(Se)]))
    i+=1
    print(i)
    
cols = ['Variance_Parameter%i'%i for i in range(1, 11)]
cols+= ['Error_Parameter%i'%i for i in range(1, 4)]
cols+= ['gnorm']
data = pd.DataFrame(estimate_error[:i], columns=cols)
data = data[~(data['gnorm']>1e-3)]
df = pd.DataFrame(data.iloc[:, :-1].stack(), columns=['value']).reset_index()

sumstats_agg = data.agg(['mean', 'min', 'median', 'max', 'std',
                         'skew', 'kurt', 'size']).T

fig, ax = plt.subplots(nrows=3)
#fig.suptitle(("Distribution of Parameter Estimation Error \n"
#             "$\\theta-\\hat{\\theta}$"))
fig.suptitle("Distribution of Parameter Estimation Error", fontweight="bold",
             fontsize=20)
ax[0].set_title("$\\theta-\\hat{\\theta}$", fontsize=16, pad=10)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.subplots_adjust(hspace=0.05, left=0.1, right=0.9, top=0.9, bottom=0.15)


g1 = sns.pointplot(x='level_1', y='value', data=df, join=False, ax=ax[0],
                   ci=99, capsize=.5)
g1.set_xticklabels([], rotation=45, ha='right')

g2 = sns.swarmplot(x='level_1', y='value', data=df, alpha=0.5, ax=ax[1],
                   size=2)
g2.set_xticklabels([], rotation=45, ha='right')

g3 = sns.boxplot(x='level_1', y='value', data=df, ax=ax[2], width=0.4)
g3.set_xticklabels(lmm.res_names, rotation=30, ha='right')
g3.set_xlabel('')
g1.axhline(0, color='k')
g2.axhline(0, color='k')
g3.axhline(0, color='k')

#at just 200 iterations this was hitting the edges of the parameter space,
#in that each of the off diagonal variance components was getting fairly close
#to the fixed diagonal components (i.e. 16 or 4), and this was removing
#models that failed to converge
print(maximum)
print(minimum)

