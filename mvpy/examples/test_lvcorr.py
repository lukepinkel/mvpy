#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:37:25 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import mvpy.api as mv # analysis:ignore
from mvpy.utils import linalg_utils, base_utils, statfunc_utils  # analysis:ignore
import seaborn as sns # analysis:ignore
sns.set_style('darkgrid')

def generate_polychoric(r=0.5, n=1000, nxc=5, nyc=5):
    S = np.eye(2)
    S[0, 1] = S[1, 0] = r
    X = sp.stats.multivariate_normal(np.zeros(2), S).rvs(n)
    df = pd.DataFrame(X, columns=['zx', 'zy'])
    df['x'] = pd.qcut(df['zx'], nxc).cat.codes.astype(float)
    df['y'] = pd.qcut(df['zy'], nxc).cat.codes.astype(float)
    return df


def generate_polyserial(r=0.5, n=1000, ncy=5):
    S = np.eye(2)
    S[0, 1] = S[1, 0] = r
    X = sp.stats.multivariate_normal(np.zeros(2), S).rvs(n)
    df = pd.DataFrame(X, columns=['zy', 'x'])
    df['y'] = pd.qcut(df['zy'], ncy).cat.codes.astype(float)
    return df




res = []
rhos = np.arange(-0.6, 0.6+0.1, 0.1)

bounds =[(-1.0+1e-16, 1.0-1e-16)]
for r in rhos:
    for i in range(500):
        dfplc = generate_polychoric(r=r)
        plc = mv.Polychoric("x", "y", dfplc)
        plc.fit()
        res.append([plc.rho_hat, plc.se_rho, r])
        print(i)
        
polychoric_res = pd.DataFrame(res, columns=['rh', 'se_r', 'r'])
polychoric_res['error'] = polychoric_res['rh'] -polychoric_res['r']
polychoric_res['error2'] = polychoric_res['error']**2

polychoric_res.groupby('r')[['error']].agg(['mean', 'median', 'std', 'skew'])
polychoric_res.groupby('r')['rh'].agg(['mean', 'median', 'std', 'skew'])
polychoric_res.groupby('r')['se_r'].agg(['mean', 'median', 'std', 'skew'])

seres = pd.concat([polychoric_res.groupby('r')['se_r'].agg(['mean']),
              polychoric_res.groupby('r')['error'].agg(['std'])], axis=1)

for x in polychoric_res['r'].unique():
    sns.distplot(polychoric_res.loc[polychoric_res['r']==x, 'rh'])

sns.jointplot('mean', 'std', seres)

res_polyserial = []
rhos = np.arange(-0.6, 0.6+0.1, 0.3)

bounds =[(-1.0+1e-16, 1.0-1e-16)]
for r in rhos:
    for i in range(500):
        dfpls = generate_polyserial(r=r)
        pls = mv.Polyserial("x", "y", dfpls)
        pls.fit()
        res_polyserial.append([pls.rho_hat, pls.se_rho, r])
        print(i)
        
polyserial_res = pd.DataFrame(res_polyserial, columns=['rh', 'se_r', 'r'])
polyserial_res['error'] = polyserial_res['rh'] - polyserial_res['r']
polyserial_res['error2'] = polyserial_res['error']**2
polyserial_res['se_r'] = np.concatenate(polyserial_res['se_r'].values)

polyserial_res.groupby('r')[['error']].agg(['mean', 'median', 'std', 'skew'])
polyserial_res.groupby('r')['rh'].agg(['mean', 'median', 'std', 'skew'])
polyserial_res.groupby('r')['se_r'].agg(['mean', 'median', 'std', 'skew'])

seres2 = pd.concat([polyserial_res.groupby('r')['se_r'].agg(['mean']),
                    polyserial_res.groupby('r')['error'].agg(['std'])], axis=1)


sns.jointplot('mean', 'std', seres2)








