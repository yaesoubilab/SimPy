import SimPy.FittingProbDist_ML as Fit
import numpy as np
import SimPy.RandomVariantGenerators as RVGs
from tests.ProbDistributions.RVGtests import get_samples


# 12 NegativeBinomial
dist = RVGs.NegativeBinomial(3, 0.3, 1)
dat_neg_bin = np.array(get_samples(dist, np.random))    # mean, sigma
dictResults=Fit.fit_negative_binomial(dat_neg_bin, 'Data', fixed_location=1, bin_width=1)
print("Fitting NegativeBinomial:", dictResults)