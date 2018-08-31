import SimPy.RandomVariantGenerators as RVGs
from SimPy.Optimization import SimModel


class Xto2(SimModel):
    # a simple simulation model that represents x^2 + noise (where noise is normally distributed)
    def __init__(self, err_sigma):
        """"
        :param err_sigma is the standard deviation of noise term
        """

        SimModel.__init__(self)
        # create a random number generator
        self._rng = RVGs.RNG(seed=1)
        # create a normal distribution to model noise
        self._err = RVGs.Normal(loc=0, scale=err_sigma)

    def get_obj_value(self, x):
        """ returns one realization from x^2+noise """

        return x*x + self._err.sample(self._rng)
