# mvpy
Currently includes rough implementations of 
- Cumulative link models for ordinal regression.  
- Factor analytic methods
  - Confirmatory factor analysis
  - Exploratory factor analysis
  - Factor rotation
- Linear mixed models capable of handling multivariate models
- Latent variable correlations for handling polychorric, polytomous and tetrachoric correlation
- Partial least squares (soft modeling) techniques
  - Partial least squares covariance
  - Partial least squares regression
    - SIMPLS
    - NIPALS
    - Wolds two block mode A (W2A)
  - Partial least squares structural equation modeling
  - Canonical correlation
  - Sparse CCA
- Structural equation modeling (Only the ML estimator works, although robust standard errors and adjusted test statistics are working)
- Simple linear models
- Generalized Linear Models 
  - Poission, Gamma, and Bernoulli distributions
  - Logit, Probit, Log, and Reciprocal links
- Negative Binomial Models
  - Currently only supports NB2, although plans exist to implement other overdispersed count models 

Although all of these have been tested against results published in the literature, or those obtained in R(in some cases transitively via statsmodels), they have not been tested systematically, and some of the code is very rough.

The cumulative link model is planned to be subsumed by a GLM module, while the factor analytic methods, latent variable correlation methods, and structural equation models are planned to be implemented under a general latent variable model. 

The partial least squares functions (generally with the exception of PLS-SEM) suffer from a variety of issues, and inconsistencies, as they were programmed with large gaps in between them, and PLSR, PLSC, and CCA may be subsumed by a more coherent model.

Math outlining the basis and details of the implementation of multivariate mixed linear models can be viewed https://www.overleaf.com/read/kwfwwnsrybtk
