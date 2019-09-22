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
- Structural equation modeling (ML)
- Simple linear models

Although all of these have been tested against results published in the literature, or those obtained in R(in some cases transitively via statsmodels), they have not been tested systematically, and some of the code is very rough.

The cumulative link model is planned to be subsumed by a GLM module, while the factor analytic methods, latent variable correlation methods, and structural equation models are planned to be implemented under a general latent variable model.  

