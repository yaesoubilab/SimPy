import scr.ProbDistFitting as Fit
import numpy as np
import scr.RandomVariantGenerators as RVGs
import scipy.stats as scs

# simulate some data
np.random.seed(1)

# 1 fitting a exponential distribution
dat_exp = np.random.exponential(5, 1000)            # generate data
dictResults=Fit.fit_exp(dat_exp, 'Data')        # fit
print("Fitting Exponential:", dictResults)






# 6 fitting a gamma distribution
dat_gamma = np.random.gamma(10, 2,1000)     # generate data
dictResults=Fit.fit_gamma(dat_gamma, 'Data')        # fit
print("Fitting Gamma:", dictResults)

# 9 fitting a JohnsonSb distribution
dat_JohnsonSb = scs.johnsonsb.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
dictResults=Fit.fit_johnsonSb(dat_JohnsonSb, 'Data')    # fit
print("Fitting johnsonSb:", dictResults)

# 10 fitting a JohnsonSu distribution
dat_JohnsonSu = scs.johnsonsu.rvs(a=10, b=3, loc=0, scale=1, size=1000)    # generate data
dictResults=Fit.fit_johnsonSu(dat_JohnsonSu, 'Data')    # fit
print("Fitting johnsonSu:", dictResults)

# 17 fitting a Weibull distribution
dat_weibull = np.random.weibull(5, 1000)    # generate data
dictResults=Fit.fit_weibull(dat_weibull, 'Data')    # fit
print("Fitting Weibull:", dictResults)

# 18 fitting a Poisson distribution
dat_poisson = np.random.poisson(30, 1000)    # generate data
dictResults=Fit.fit_poisson(dat_poisson, 'Data')    # fit
print("Fitting Poisson:", dictResults)