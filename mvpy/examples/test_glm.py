#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 16:34:16 2020

@author: lukepinkel
"""

import numpy as np # analysis:ignore
import scipy as sp # analysis:ignore
import scipy.stats # analysis:ignore
import pandas as pd # analysis:ignore
import mvpy.api as mv# analysis:ignore
from mvpy.utils import linalg_utils, base_utils # analysis:ignore
import statsmodels.api as sm # analysis:ignore


spector_data = sm.datasets.spector.load_pandas()
spector_data.exog = sm.add_constant(spector_data.exog)
spector_data = pd.concat([spector_data.exog, spector_data.endog], axis=1)


poisson_data = pd.read_csv("https://stats.idre.ucla.edu/stat/data/poisson_sim.csv")
poisson_data.index = poisson_data['id']

fair_data = sm.datasets.fair.load_pandas().data

S = mv.vine_corr(10, 20)
X = mv.multi_rand(S)
b = np.random.randint(-5, 5, size=(10))*1.0
b/= 10.0
w = np.random.randint(1, 20, size=(1000))*1.0
eta = X.dot(b)
mu = mv.Binomial(mv.LogitLink, weights=w).inv_link(eta)
y = sp.stats.binom(w.astype(int), mu).rvs()
binomial_data = pd.DataFrame(np.hstack([X, y[:, None]]))
binomial_data.columns = ['x%i'%i for i in range(1, 11)]+['y']
binomial_data['y'] = binomial_data['y'] / w


scot_data = sm.datasets.scotland.load(as_pandas=False)
scot_data.exog = sm.add_constant(scot_data.exog)
scot_df = pd.DataFrame(scot_data.data)


S, v, b = mv.vine_corr(10, 20), 0.5, np.random.randint(-5, 5, size=(10))/10.0
X = mv.multi_rand(S)
b = np.append(1, b)
X = np.concatenate([np.ones((1000, 1)), X], axis=1)
eta, w = X.dot(b), np.random.randint(1, 20, size=(1000))*1.0
mu = mv.NegBinom().inv_link(eta)

p = v / (mu + v)
y = sp.stats.nbinom(v, p).rvs()

negbin_data = pd.DataFrame(np.hstack([X, y[:, None]]))
negbin_data.columns = ['x%i'%i for i in range(1, 12)]+['y']


S = mv.vine_corr(10, 20)
X = mv.multi_rand(S)
b = np.random.randint(-5, 5, size=(10))*1.0
b/= 10.0
eta = X.dot(b)
s2 = 3
mu = mv.Gaussian(mv.IdentityLink).inv_link(eta)
y = sp.stats.norm(mu, s2).rvs()
normal_data = pd.DataFrame(np.hstack([X, y[:, None]]))
normal_data.columns = ['x%i'%i for i in range(1, 11)]+['y']



S = mv.vine_corr(20, 50)
X = mv.multi_rand(S, 150000)
b = np.random.randint(-5, 5, size=(20))*1.0
b/= 10.0
eta = X.dot(b)

mu = mv.Binomial().inv_link(eta)
y = sp.stats.binom(1, mu).rvs()
logit_data = pd.DataFrame(np.hstack([X, y[:, None]]))
logit_data.columns = ['x%i'%i for i in range(1, 21)]+['y']



modf = mv.GLM("y~x2+x3+x4+x5+x6+x7+x8+x9+x10+x11", data=negbin_data, 
              fam=mv.NegBinom(mv.LogLink), scale_estimator='fixed')


modnr = mv.GLM("y~x2+x3+x4+x5+x6+x7+x8+x9+x10+x11", data=negbin_data, 
              fam=mv.NegBinom(mv.LogLink), scale_estimator='NR')

modf.fit()
modnr.fit()


smd = sm.formula.glm("y~x2+x3+x4+x5+x6+x7+x8+x9+x10+x11", data=negbin_data, 
             family=sm.families.NegativeBinomial()).fit()

np.allclose(modf.gradient(modf.params+0.1), mv.fprime(modf.loglike, modf.params+0.1))


gamma_mod = mv.GLM(scot_data.endog_name+"~"+"+".join(scot_data.exog_name), 
                    data=scot_df, fam=mv.Gamma(mv.ReciprocalLink), 
                    scale_estimator='NR')

gamma_mod.fit('mn')




gamma_mod = mv.GLM(scot_data.endog_name+"~"+"+".join(scot_data.exog_name), 
                    data=scot_df, fam=mv.Gamma(), 
                    scale_estimator='M')

gamma_mod.fit('mn')



gamma_mod = mv.GLM(scot_data.endog_name+"~"+"+".join(scot_data.exog_name), 
                    data=scot_df, fam=mv.Gamma(mv.LogLink), 
                    scale_estimator='M')

gamma_mod.fit('sp')


gauss_mod = mv.GLM("y~x1+x2+x3+x4+x5+x6+x7+x8+x9+x10",
                    data=normal_data, fam=mv.Gaussian(mv.IdentityLink), 
                    scale_estimator='NR')

gauss_mod.fit('sp')


logit_mod = mv.GLM("GRADE~GPA+TUCE+PSI", data=spector_data,
                    fam=mv.Binomial())

logit_mod.fit()



logit_mod = mv.GLM("y~-1+"+"+".join(logit_data.columns[:-1]), data=logit_data,
                    fam=mv.Binomial())

logit_mod.fit()


