import scr.ProbDistFitting as Fit
import numpy as np

# simulate some data
np.random.seed(1)

# fitting a gamma distribution
dat_gamma = np.random.gamma(10, 2,1000)     # generate data
dictResults=Fit.fit_gamma(dat_gamma, 'Data')        # fit
print("Fitting Gamma:", dictResults)

# fitting a Poisson distribution
dat_poisson = np.random.poisson(30, 1000)    # generate data
dictResults=Fit.fit_poisson(dat_poisson, 'Data')    # fit
print("Fitting Poisson:", dictResults)