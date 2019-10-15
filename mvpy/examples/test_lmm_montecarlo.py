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
import matplotlib.gridspec as gridspec#analysis:ignore

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
                   ci=99.99, capsize=.5, n_boot=10000)
g1.set_xticklabels([], rotation=45, ha='right')

g2 = sns.swarmplot(x='level_1', y='value', data=df, alpha=0.5, ax=ax[1],
                   size=1)
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





fig = plt.figure()
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
gs = gridspec.GridSpec(nrows=5, ncols=3)

ax0 = fig.add_subplot(gs[0, 0])
sns.distplot(data.iloc[:, 0], ax=ax0)


ax1 = fig.add_subplot(gs[0, 1], sharex=ax0, sharey=ax0)
sns.distplot(data.iloc[:, 1], ax=ax1)

ax2 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
sns.distplot(data.iloc[:, 2], ax=ax2)

ax3 = fig.add_subplot(gs[1, 0], sharex=ax2, sharey=ax2)
sns.distplot(data.iloc[:, 3], ax=ax3)

ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax3)
sns.distplot(data.iloc[:, 4], ax=ax4)

ax5 = fig.add_subplot(gs[1, 2], sharex=ax4, sharey=ax4)
sns.distplot(data.iloc[:, 5], ax=ax5)

ax6 = fig.add_subplot(gs[2, 0], sharex=ax5, sharey=ax5)
sns.distplot(data.iloc[:, 6], ax=ax6)

ax7 = fig.add_subplot(gs[2, 1], sharex=ax6, sharey=ax6)
sns.distplot(data.iloc[:, 7], ax=ax7)

ax8 = fig.add_subplot(gs[2, 2], sharex=ax7, sharey=ax7)
sns.distplot(data.iloc[:, 8], ax=ax8)

ax9 = fig.add_subplot(gs[3, 0], sharex=ax8, sharey=ax8)
sns.distplot(data.iloc[:, 9], ax=ax9)

ax10 = fig.add_subplot(gs[3, 1], sharex=ax9, sharey=ax9)
sns.distplot(data.iloc[:, 10], ax=ax10)

ax11 = fig.add_subplot(gs[3, 2], sharex=ax10, sharey=ax10)
sns.distplot(data.iloc[:, 11], ax=ax11)

ax12 = fig.add_subplot(gs[4, 1], sharex=ax11, sharey=ax11)
sns.distplot(data.iloc[:, 12], ax=ax12)





