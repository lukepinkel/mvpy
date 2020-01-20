#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:16:52 2020

@author: lukepinkel
"""
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import mvpy.api as mv
from mvpy.utils import data_utils
import seaborn as sns
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
np.set_printoptions(suppress=True)

# Generative model 1

np.random.seed(3)  
n_units=150
n_unit_obs=10
n_levels=2
n_level_effects=2
beta_params_v=5
beta_params_e=30

Sv = mv.vine_corr(n_levels*n_level_effects, beta_params_v)
Se = mv.vine_corr(n_levels, beta_params_e)

Wv = np.eye(n_units)
We = np.eye(n_units*n_unit_obs)

Vc = np.kron(Sv, Wv)
Ve = np.kron(Se, We)
Z = sp.stats.norm(0, 1).rvs(n_units*n_unit_obs)
Zi = np.hstack([np.ones((Z.shape[0], 1)), Z[:, None]])
J = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
J = data_utils.dummy_encode(J, complete=True)
Z = mv.khatri_rao(J.T, Zi.T).T
Z = np.kron(np.eye(2), Z)
beta = np.random.normal(size=(2, 1))
X = np.kron(np.ones((2, 1)), Zi)

U = mv.center(mv.multi_rand(Vc))
E = mv.center(mv.multi_rand(Ve, size=Ve.shape[1]*2))
e = E[[0]].T
u = U[[0]].T

y = X.dot(beta)+Z.dot(u)+e
x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])

data = np.concatenate([y.reshape(n_units*n_unit_obs, 2, order='F'), Zi[:, [1]]], axis=1)
data = pd.DataFrame(data, columns=['y1', 'y2', 'x1'])
data['id'] = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
fixed_effects = "~x1+1"
random_effects = {"id":"~x1+1"}
yvar = ['y1', 'y2']



model = mv.LMM(fixed_effects, random_effects,  yvar, data)
model.fit()
print(mv.corr(mv.kmat(300, 2).dot(u), model.u))

sns.jointplot(mv.kmat(300, 2).dot(u), model.u, stat_func=sp.stats.pearsonr)
#model.fit(hess_opt=True)

df = sm.datasets.get_rdataset("sleepstudy", "lme4").data

lme = mv.LMM("1+Days", {"Subject":"~1+Days"}, "Reaction", data=df)
lme.fit(hess_opt=True)

#Via lme4
#Intercept x Intercept - 611.90
#Intercept x Days      - 9.67
#Days x Days           - 35.08
#Error Variance        - 654.94
#Intercept             - 251.40
#Days                  - 10.467

df['Subject2'] = df['Subject']
lme = mv.LMM("1+Days", {"Subject":"~1", "Subject2":"Days-1"}, "Reaction", data=df)
lme.fit(hess_opt=True)

#Alternative Eq generative model

np.random.seed(3)  
n_units=150
n_unit_obs=10
n_levels=2
n_level_effects=2
beta_params_v=5
beta_params_e=30

Sv = mv.vine_corr(n_levels*n_level_effects, beta_params_v)
Se = mv.vine_corr(n_levels, beta_params_e)

Wv = np.eye(n_units)
We = np.eye(n_units*n_unit_obs)

Vc = np.kron(Wv, Sv)
Ve = np.kron(We, Se)

Z = sp.stats.norm(0, 1).rvs(n_units*n_unit_obs)

Zi = np.hstack([np.ones((Z.shape[0], 1)), Z[:, None]])
J = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
J = data_utils.dummy_encode(J, complete=True)
Z = mv.khatri_rao(J.T, Zi.T).T
Z = np.kron(np.eye(2), Z)
beta = np.random.normal(size=(2, 1))
X = np.kron(np.ones((2, 1)), Zi)

u = sp.stats.multivariate_normal(np.zeros(Vc.shape[0]), Vc).rvs(1)
e = sp.stats.multivariate_normal(np.zeros(Ve.shape[0]), Ve).rvs(1)
u, e = np.atleast_2d(u).T, np.atleast_2d(e).T

y = X.dot(beta)+Z.dot(u)+e
x = np.concatenate([np.arange(n_unit_obs) for i in range(n_units)])

data = np.concatenate([y.reshape(n_units*n_unit_obs, 2, order='F'), Zi[:, [1]]], axis=1)
data = pd.DataFrame(data, columns=['y1', 'y2', 'x1'])
data['id'] = np.concatenate([mv.jmat(n_unit_obs)*i for i in range(n_units)])
fixed_effects = "~x1+1"
random_effects = {"id":"~x1+1"}
yvar = ['y1', 'y2']



model = mv.LMM(fixed_effects, random_effects,  yvar, data)
model.fit()
print(mv.corr(mv.kmat(300, 2).dot(u), model.u))

sns.jointplot(mv.kmat(300, 2).dot(u), model.u, stat_func=sp.stats.pearsonr)
#model.fit(hess_opt=True)

df = sm.datasets.get_rdataset("sleepstudy", "lme4").data

lme = mv.LMM("1+Days", {"Subject":"~1+Days"}, "Reaction", data=df)
lme.fit(hess_opt=True)



nmz, ndz, h2 = 1400, 1400, 0.7
n_obs = (nmz+ndz)*2
n_groups = nmz+ndz
vp = 2
e_c = 0.5

a = h2*vp
ce = (1-h2)*vp
e = ce / (1.0 + e_c)
c = ce - e

prop_mz_reared_apart = 0.5
prop_dz_reared_apart = 0.5

nmra = int(nmz*prop_mz_reared_apart)
ndra = int(ndz*prop_dz_reared_apart)

nmrt = int(nmz-nmra)
ndrt = int(ndz-ndra)


M0 = np.eye(2)
M1 = np.ones((2, 2))
M2 = np.eye(2) + 0.50 * np.flip(np.eye(2), axis=1)
M4 = np.eye(2) + 0.25 * np.flip(np.eye(2), axis=1)

A = sp.linalg.block_diag(*([M1]*nmz+[M2]*ndz))
C = sp.linalg.block_diag(*([M1]*nmrt+[M0]*nmra+[M1]*ndrt+[M0]*ndra))
E = np.eye(n_obs)


S = a * A + c * C + e * E

ua = sp.stats.multivariate_normal(np.zeros(A.shape[0]), np.eye(A.shape[0])).rvs()
uv = mv.chol(a*A).dot(ua)
ue = sp.stats.multivariate_normal(np.zeros(A.shape[0]), c * C + e * E).rvs()

u = uv+ue

#u = sp.stats.multivariate_normal(np.zeros(S.shape[0]), S).rvs()

z1 = np.concatenate([np.tile('M', int(nmz)), np.tile('F', int(nmz)),
                    np.tile('M', int(ndz)), np.tile('F', int(ndz))])
z2 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.2, 0.4, 0.3, 0.1])
z2 = np.kron(z2, np.ones(2))

z3 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.25, 0.25, 0.25, 0.25])
z3 = np.kron(z3, np.ones(2))


df = pd.DataFrame(u, columns=['u'])
df['z1'] = z1
df['z2'] = z2
df['z3'] = z3
df['z3'] = df['z3'].replace(dict(zip([0, 1, 2, 3], ['A1', 'A2', 'A3', 'A4'])))

D = patsy.dmatrix("C(z3, Treatment(reference='A2'))+C(z2)+C(z1)", data=df, return_type='dataframe', eval_env=True)
df['y'] = df['u']+D.dot(np.array([-0.1, 0.3, -1.0, -0.6, -0.2, 0.1, -0.1, 0.3]))
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
params = []
nmz, ndz, h2 = 50, 50, 0.7
n_obs = (nmz+ndz)*2
n_groups = nmz+ndz
vp = 2
e_c = 0.5

a = h2*vp
ce = (1-h2)*vp
e = ce / (1.0 + e_c)
c = ce - e

prop_mz_reared_apart = 0.5
prop_dz_reared_apart = 0.5

nmra = int(nmz*prop_mz_reared_apart)
ndra = int(ndz*prop_dz_reared_apart)

nmrt = int(nmz-nmra)
ndrt = int(ndz-ndra)


M0 = np.eye(2)
M1 = np.ones((2, 2))
M2 = np.eye(2) + 0.50 * np.flip(np.eye(2), axis=1)
M4 = np.eye(2) + 0.25 * np.flip(np.eye(2), axis=1)

A = sp.linalg.block_diag(*([M1]*nmz+[M2]*ndz))
C = sp.linalg.block_diag(*([M1]*nmrt+[M0]*nmra+[M1]*ndrt+[M0]*ndra))
E = np.eye(n_obs)


S = a * A + c * C + e * E

for i in range(1500):
    ua = sp.stats.multivariate_normal(np.zeros(A.shape[0]), np.eye(A.shape[0])).rvs()
    uv = mv.chol(a*A).dot(ua)
    ue = sp.stats.multivariate_normal(np.zeros(A.shape[0]), c * C + e * E).rvs()
    
    u = uv+ue
    
    #u = sp.stats.multivariate_normal(np.zeros(S.shape[0]), S).rvs()
    
    z1 = np.concatenate([np.tile('M', int(nmz)), np.tile('F', int(nmz)),
                        np.tile('M', int(ndz)), np.tile('F', int(ndz))])
    z2 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.2, 0.4, 0.3, 0.1])
    z2 = np.kron(z2, np.ones(2))
    
    z3 = np.random.choice(np.arange(0, 4), n_groups, replace=True, p=[0.25, 0.25, 0.25, 0.25])
    z3 = np.kron(z3, np.ones(2))
    
    
    df = pd.DataFrame(u, columns=['u'])
    df['z1'] = z1
    df['z2'] = z2
    df['z3'] = z3
    df['z3'] = df['z3'].replace(dict(zip([0, 1, 2, 3], ['CatA', 'CatB', 'CatC', 'CatD'])))
    
    D = patsy.dmatrix("C(z3, Treatment(reference='CatB'))+C(z2)+C(z1)", data=df, return_type='dataframe', eval_env=True)
    df['y'] = df['u']+D.dot(np.array([-0.1, 0.3, -1.0, -0.6, -0.2, 0.1, -0.1, 0.3]))
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
    
    
    fe = "C(z3, Treatment(reference='CatB'))+C(z2)+C(z1)"
    lmm = mv.LMM(fe, {"pid":"~x1-1", "tid1":"~x2-1", "tid2":"~1"}, "y", data=df)
    lmm.fit(hess_opt=True, verbose=0)
    params.append({'theta':lmm.params, '|g|':lmm.gnorm})
    print(i)
    
param_df = np.concatenate([param['theta'][:, None] for param in params if param['|g|']<1e-4], axis=1).T
param_df = np.concatenate([param[:, None] for param in params], axis=1).T

heritability = [hfunc(param) for param in param_df]
heritability
sns.distplot(heritability)
theta = lmm.params
v = v = dfunc(theta)
h2_se =  np.sqrt(np.dot(v.T,  lmm.hessian_inv).dot(v))

np.std(heritability)
param_data = pd.DataFrame(np.concatenate([np.concatenate([param['theta'][:, None], 
                                                          np.array([[param['|g|']]])]) 
                        for param in params], axis=1).T, columns=['t1', 't2', 't3', 't4', 'g'])
param_data['h2'] = [hfunc(param) for param in np.concatenate([param['theta'][:, None] for param in params], axis=1).T]
param_data['converged'] = param_data['g']<1e-4


sns.distplot(param_data.loc[param_data['g']<1e-4, 'h2'])



fig = plt.figure()
gs = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)
ax_marg = fig.add_subplot(gs[-1, :])
ax_main = fig.add_subplot(gs[:-1], sharex=ax_marg)
plt.setp(ax_main.get_xticklabels(), visible=False)
plt.setp(ax_marg.get_yticklabels(), visible=False)

ax_marg.boxplot(param_data.loc[param_data['g']<1e-4, 'h2'], vert=False,
                widths=0.5, patch_artist=True, 
                boxprops=dict(facecolor=sns.color_palette()[0], alpha=0.5),
                )

g1 = sns.distplot(param_data.loc[param_data['g']<1e-4, 'h2'], ax=ax_main)
g2 =sns.distplot(sp.stats.norm(0.7, h2_se).rvs(2000000), ax=ax_main)

ax_main.set_xlabel("")
fig.suptitle(("$F(\hat{\\sigma_{h}}^{2})\\approx"
              "\\left[\\frac{\\partial\\sigma_{h}^{2}}{\\partial\\theta}\\right]"
              "\\frac{\\partial^{2}\ell(\\theta)}{(\\partial\\theta)(\\partial\\theta')}"
              "\\left[\\frac{\\partial\\sigma_{h}^{2}}{\\partial\\theta}\\right]^{'}$"),
             fontsize=25)


ax_main.legend([g1.legend(), g2.legend()], labels=["Bootstrap", "Parametric"])

mng = plt.get_current_fig_manager()
mng.window.showMaximized()






