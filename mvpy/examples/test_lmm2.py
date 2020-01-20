#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:46:29 2019

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import mvpy.api as mv # analysis:ignore
import patsy # analysis:ignore
import seaborn as sns# analysis:ignore

# Specify number of monozygotic, dizygotic, heritability, phenotypic var, and 
# error var; for large numbers, this can be slow
nmz, ndz, h2, vp, e_c = 1400, 1400, 0.7, 2, 0.5
n_groups, n_obs = nmz+ndz, (nmz+ndz)*2

# compute genetic var, shared env, and env var
a, ce = h2*vp, (1-h2)*vp
e = ce / (1.0 + e_c)
c = ce - e

# specify number of twins reared together versus apar 
prop_mz_reared_apart, prop_dz_reared_apart = 0.5, 0.5
nmra, ndra = int(nmz*prop_mz_reared_apart), int(ndz*prop_dz_reared_apart)
nmrt, ndrt = int(nmz-nmra), int(ndz-ndra)


# Generate matrices for respective covariacne structures
M0 = np.eye(2)
M1 = np.ones((2, 2))
M2 = np.eye(2) + 0.50 * np.flip(np.eye(2), axis=1)
M4 = np.eye(2) + 0.25 * np.flip(np.eye(2), axis=1)

# Compute covariance matrices for each source of variance
A = sp.linalg.block_diag(*([M1]*nmz+[M2]*ndz))
C = sp.linalg.block_diag(*([M1]*nmrt+[M0]*nmra+[M1]*ndrt+[M0]*ndra))
E = np.eye(n_obs)

# Obtain total phenotypuc covariance 
S = a * A + c * C + e * E

# Get random effects
ua = sp.stats.multivariate_normal(np.zeros(A.shape[0]), np.eye(A.shape[0])).rvs()
uv = mv.chol(a*A).dot(ua)
ue = sp.stats.multivariate_normal(np.zeros(A.shape[0]), c * C + e * E).rvs()

u = uv+ue

#u = sp.stats.multivariate_normal(np.zeros(S.shape[0]), S).rvs()

# Add covariate for sex
z1 = np.concatenate([np.tile('M', int(nmz)), np.tile('F', int(nmz)),
                    np.tile('M', int(ndz)), np.tile('F', int(ndz))])
# Add unbalanced categorical covariate 
z2 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.2, 0.4, 0.3, 0.1])
z2 = np.kron(z2, np.ones(2))
# Add balanced categorical covariate
z3 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.25, 0.25, 0.25, 0.25])
z3 = np.kron(z3, np.ones(2))

# Combine covariates in dataframe
df = pd.DataFrame(u, columns=['u'])
df['z1'] = z1
df['z2'] = z2
df['z3'] = z3
df['z3'] = df['z3'].replace(dict(zip([0, 1, 2, 3], ['A1', 'A2', 'A3', 'A4'])))

D = patsy.dmatrix("C(z3, Treatment(reference='A2'))+C(z2)+C(z1)", data=df, return_type='dataframe', eval_env=True)
df['y'] = df['u']+D.dot(np.array([-0.1, 0.3, -1.0, -0.6, 
                                  -0.2, 0.1, -0.1, 0.3]))
df['pid'] = np.arange(n_obs)
df['tid1'] = np.kron(np.arange(n_groups), np.ones(2))
df['tid2'] = np.concatenate([np.kron(np.arange(nmrt), np.ones(2)),
                             np.arange(nmrt, nmrt+nmra*2),
                             np.kron(np.arange(nmrt+nmra*2, nmrt+nmra*2+ndrt), np.ones(2)),
                             np.arange(nmrt+nmra*2+ndrt, nmrt+nmra*2+ndrt+ndra*2)])
df['M'] = np.concatenate([np.ones(n_groups), np.zeros(n_groups)])
df['Mbar'] = 1-np.concatenate([np.ones(n_groups), np.zeros(n_groups)])

df['x1'] = np.sqrt(1.0/2.0)*df['Mbar']
df['x2'] = np.sqrt(1.0/2.0)*df['Mbar']+df['M']
df['twin'] = np.kron(np.ones(n_groups), np.arange(2))


fe = "C(z3, Treatment(reference='A2'))+C(z2)+C(z1)"
lmm = mv.LMM(fe, {"pid":"~x1-1", "tid1":"~x2-1", "tid2":"~1"}, "y", data=df)
lmm.fit(hess_opt=True)
 # if optimization is too slow, gradient descent is usually pretty good

"""
%timeit lmm.loglike(lmm.theta)
%timeit lmm.gradient(lmm.theta)
%timeit lmm.hessian(lmm.theta)
"""
sns.set_style('darkgrid')
sns.jointplot(lmm.Z.dot(lmm.u)[:, 0], u, stat_func=sp.stats.pearsonr,
              kind='reg')

theta = lmm.params

np.sum(theta[0]+theta[1])/np.sum(theta)

def hfunc(theta):
    u = np.zeros(4)
    u[:2] = 0.5
    v = np.ones(4)
    v[:2]/=2
    
    return np.dot(theta.T, u)/np.dot(theta.T, v)

def dfunc(theta):
    u = np.zeros(4)
    u[:2] = 0.5
    v = np.ones(4)
    v[:2]/=2
    
    t = np.dot(theta.T, v)
    d = 1.0 / t * u - (1.0 / (t**2)) * np.dot(theta.T, u) * v
    return d

v = dfunc(theta)
h2_se =  np.sqrt(np.dot(v.T,  lmm.hessian_inv).dot(v))
res = lmm.res.copy()
res.loc['h2'] = [hfunc(theta), h2_se, hfunc(theta)/h2_se, 
       sp.stats.norm.sf(hfunc(theta)/h2_se)*2]
res['Cl-'] = -1.96*res['Standard Error']+res['Parameter Estimate']
res['Cl+'] = 1.96*res['Standard Error']+res['Parameter Estimate']

res = res[res.columns[[0, 1, 5, 4, 2, 3]]]

