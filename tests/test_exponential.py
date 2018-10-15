from unittest import TestCase
from SimPy import RandomVariantGenerators as RVGs


class TestExponential(TestCase):
    def test_sample(self):

        exp_dist = RVGs.Exponential(scale=1)

        exp_dist.sample()
        self.fail()
