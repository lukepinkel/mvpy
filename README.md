# mvpy
Currently includes rough implementations of 
- Cumulative link models for ordinal regression.  
- Factor analytic methods. 
  - Confirmatory factor analysis via the CFA class, fit through EM
  - Exploratory factor analysis via the EFA class, fit through EM or Lawleys ML algorithm
  - Both via the FactorAnalysis class, fit through constrained Newtons method using a parameterization more robust to small unique variances
  - Factor rotation
- Linear mixed models capable of handling multivariate models.  Note that the p-values presented in the results table should    not be taken seriously, as they are computed under the assumption of (n-p) degrees of freedom (n observations minus p features).
- Generalized linear mixed models via the penalized quasi-likelihood method.
- Latent variable correlations for handling polychorric, polytomous and tetrachoric correlation
- Partial least squares (soft modeling) techniques
  - Partial least squares covariance
  - Partial least squares regression
    - SIMPLS
    - NIPALS
    - Wolds two block mode A (W2A)
  - Partial least squares structural equation modeling
  - Canonical correlation
  - Sparse Canonical Correlation 
- Structural equation modeling using ML, and GLS with normal, wishart, and adf weight matrices
- Linear models that implement a variety of univariate and multivariate hypothesis tests, and can implement MANOVA.
- Robust linear regression with Hubers T, Tukeys Bisquare (Biweight), and Hampels function.
- Generalized Linear Models 
  - Supports Gaussian, Binomial, Gamma, Gaussian, Inverse Gaussian, Poisson and Negative Binomial distributions
  - Supports Cloglog, Logit, Log, Log Complement, Probit, Negative Binomial and Reciprocal links.
- Negative Binomial Models
  - Currently only supports NB2, although plans exist to implement other overdispersed count models 
- Random correlation matrix generation via the vine method, onion method, or factor method
- Multivariate non-normal data with the ability to specify (standardized) third and fourth order moments. 
## Speed
For most models, internal optimization is done using scipy's trust-constr, which is robust but fairly slow.  All models have an option to pass to another choice to the optimizer; a safe and quick alternative to use is trust-ncg. 
## Testing and Validity
Although all of these have been tested against results published in the literature, or those obtained in R(in some cases transitively via statsmodels), they have not been tested systematically, and some of the code is very rough.
