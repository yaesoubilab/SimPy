from tests.ProbDistributions import RVGtests as Tests
import SimPy.RandomVariantGenerators as rndSupport

rng = rndSupport.RNG(1)
Tests.test_non_homogeneous_exponential(rng, rates=[1, 2])
