#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 02:42:44 2019

@author: lukepinkel
"""
       
import itertools
import numpy as np
import matplotlib as mpl
import pandas as pd
import scipy as sp
import scipy.stats
import collections
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from ..utils import linalg_utils

def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mpl.colors.LinearSegmentedColormap('CustomMap', cdict)


def cmap_params(data, vmin=None, vmax=None, cmap=None, center=None):
        data = data[~np.isnan(data)]
        if vmin is None:
            vmin = np.percentile(data, 2) 
        if vmax is None:
            vmax = np.percentile(data, 98) 

        if cmap is None:
            if center is None:
                cmap = plt.cm.bwr
            else:
                cmap = plt.cm.inferno
        elif isinstance(cmap, str):
            cmap = mpl.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            cmap = mpl.colors.ListedColormap(cmap)
        else:
            cmap = cmap
        if center is not None:
            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            cmap = mpl.colors.ListedColormap(cmap(cc))
        return vmin, vmax, cmap

def custom_bwr_five(bmax=0.2, wmin=0.4, wmax=0.6, rmin=0.8):
    c = mpl.colors.ColorConverter().to_rgb    
    cmap = make_colormap([c("blue"), c("blue"), bmax,
                          c("blue"), c("white"), wmin,
                          c("white"),c("white"), wmax,
                          c("white"), c("red"), rmin,
                          c("red"), c("red")])
    return cmap

def custom_bwr_three(wmin=0.4, wmax=0.6):
    c = mpl.colors.ColorConverter().to_rgb    
    cmap = make_colormap([c("blue"), c("white"), wmin,
                          c("white"),c("white"), wmax,
                          c("white"), c("red")])
    return cmap

def hmap(data, cmap=None, vmin=None, vmax=None, center=None, grid=True):
    if cmap is None:
        cmap = custom_bwr_five(0.15, 0.47, 0.53, 0.85)
    vmin, vmax, cmap = cmap_params(data, cmap=cmap, center=center, vmin=vmin,
                                   vmax=vmax)
    fig, ax = plt.subplots()
    cbl = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    cbar = plt.colorbar(cbl)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    ax.grid(False)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    
    try:  
        ax.set_xticklabels(data.columns, fontsize=5, rotation=45, ha='right')
        ax.set_yticklabels(data.index, fontsize=5)
    except AttributeError:
         ax.set_xticklabels(np.arange(data.shape[1]), fontsize=5, rotation=45, ha='right')
         ax.set_yticklabels(np.arange(data.shape[1]), fontsize=5)
         
    #ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    #ax.set_yticks(np.arange(-0.6, data.shape[0]), minor=True)
    #ax.grid(which='minor', color='w', linestyle='-', linewidth=.5)
    ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0]), minor=True)
    if grid:
        ax.grid(which='minor')
        ax.set_frame_on(False)

    plt.subplots_adjust(left=0.1, top=0.95, bottom=0.15, right=1.1)
    return fig, ax, cbar
       


def param_plot(params, capsize=6, capthick=2, fmt='none', ci=95.0, 
               style='seaborn', label_rotate=30, group_sort=None, 
               sort=True, size=30, ha='center'):
    
    est = params.iloc[:, 0]
    if sort is True:
        if group_sort is not None:
            est_df = pd.DataFrame(est)
            est_df['groups'] = group_sort
            grouping = est_df.groupby(['groups'])
            est_df = grouping.apply(lambda x: x.sort_values(params.columns[0],
                                                           ascending=False))
            ix = est_df.index.get_level_values(1)
            est = est.loc[ix]
        else:
            est = est.sort_values()
    var = params.iloc[:, 1]
    x = np.arange(len(params))
    conf_ints = var*sp.stats.norm.ppf(ci/100.0)
    plt.style.use(style)
    fig, ax = plt.subplots()
    ax.errorbar(x, est, yerr=conf_ints, capsize=capsize, fmt=fmt, capthick=capthick)
    ax.axhline(0, color='black', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(est.index, rotation=label_rotate, ha=ha)
    ax.scatter(x, est, s=size)
    ax.set_title('Parameter Point Estimates and %i%% CIs'%ci)
    return fig, ax

def loadings_plot(loadings):
    ldict_r = [x.split('~') for x in loadings]
    ldict = collections.defaultdict(list)
    for v, k in ldict_r: ldict[k].append(v)
    dst = 1/(len(ldict.keys())+1)
    f_heights = np.arange(0, 1, dst)[1:]
    f_heights = dict(zip(ldict.keys(), f_heights))
    fig, ax = plt.subplots()
    factor_boxes = {}
    nested_boxes = {}
    for factor in ldict.keys():
        c_coord = f_heights[factor]
        an1 = ax.annotate(factor, xy=(0.8, c_coord), xycoords="data",
                  va="top", ha="center", bbox=dict(boxstyle="round", fc="w",
                                                      ))
        factor_boxes[factor] = an1
        #ext = ax.transData.inverted().transform(an1.get_window_extent())
        #h = ext[1, 1] - ext[0, 1]
        lower = c_coord - dst/2
        upper = c_coord + dst/2
        n_inds = len(ldict[factor])
        y_inds = np.arange(lower, upper, dst/(n_inds+1))[1:]
        y_inds = dict(zip(ldict[factor], y_inds))
        nbox = []
        for indicator in ldict[factor]:
            y = y_inds[indicator]
            an2 = ax.annotate(indicator, xy=(0.3, y), xycoords=an1,# (1,0.5) of the an1's bbox
                  xytext=(0.3, y), textcoords="figure fraction",
                  va="top", ha="left",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="->", fc="black", shrinkB=10))
            nbox.append(an2)
        nested_boxes[factor] = nbox
    return fig, ax



def regline(model):
    exog = model.model.exog[:, [1]]
    exog = linalg_utils._check_2d(linalg_utils._check_np(exog))
    min_, max_ = exog.min(), exog.max()
    G = np.linalg.inv(np.matmul(exog.T, exog))
    x = linalg_utils._check_2d(np.linspace(min_, max_, 500))
    ci = np.sqrt(np.diag(np.dot(np.dot(x, G), x.T))*model.scale)
    ci*= sp.stats.norm.ppf(.90)
    xvals = pd.DataFrame(sm.add_constant(x), columns=model.model.exog_names)
    yhat = model.predict(xvals)
    return x, yhat, ci
    
    

def lmplot(x, y, data, hue=None, figax=None, scatter_kws=None, c=None,
           cmap=plt.cm.coolwarm):
    if scatter_kws is None:
        scatter_kws={'alpha':0.5}
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    frm = y+"~"+x
    if hue is not None:
        grouper_obj = data.groupby(hue)
        groups = [grouper_obj.get_group(z) for z in grouper_obj.groups]
        models = [sm.formula.ols(frm, data=group).fit() for group in groups]
        xlines, ylines, cis = zip(*[regline(mod) for mod in models])
    else:
        groups = [data]
        models = [sm.formula.ols(frm, data=data).fit()]
        xlines, ylines, cis = zip(*[regline(mod) for mod in models])
    palette = itertools.cycle(sns.color_palette())
    for xl, yl, z, g in list(zip(xlines, ylines, cis, groups)):
        xl = linalg_utils._check_1d(xl)
        yl =  linalg_utils._check_1d(yl)
        z = linalg_utils._check_1d(z)
        color = next(palette)
        if c is None:
            skcolor = color
        else:
            skcolor = c
        ax.plot(xl, yl, color=color)
        ax.fill_between(xl, yl-z, yl+z, alpha=0.3, color=color)
        ax.scatter(g[x], g[y], color=skcolor, cmap=cmap, **scatter_kws)
    return fig, ax

    
    