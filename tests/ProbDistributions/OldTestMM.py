import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples


# simulate some data
np.random.seed(1)

# 17 fitting a Weibull distribution
dist = RVGs.Weibull(5, 1, 2)
dat_weibull = np.array(get_samples(dist, np.random))    # generate data
dictResults = RVGs.Weibull.fit_mm(np.mean(dat_weibull), np.std(dat_weibull), fixed_location=1)    # fit
print("Fitting Weibull:", dictResults)


