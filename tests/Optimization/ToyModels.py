import SimPy.RandomVariantGenerators as RVGs
from SimPy.Optimization import SimModel


class Xto2(SimModel):
    # a simple simulation model that represents x0^2 + x1^2 + noise (where noise is normally distributed)
    def __init__(self, err_sigma):
        """"
        :param err_sigma is the standard deviation of noise term
        """

        SimModel.__init__(self)

        # create a normal distribution to model noise
        self._err = RVGs.Normal(loc=0, scale=err_sigma)

    def get_obj_value(self, x, seed_index=0):
        """ returns one realization from x^2+noise """
        # create a random number generator
        rng = RVGs.RNG(seed=seed_index)

        return (x[0]+1)*(x[0]+1) + x[1]*x[1] + self._err.sample(rng)


class Xto2Constrained(SimModel):
    # a simple simulation model that represents x0^2 + x1^2 + noise (where noise is normally distributed)
    # and x1 should be greater than 1
    def __init__(self, err_sigma, penalty):
        """"
        :param err_sigma is the standard deviation of noise term
        """

        SimModel.__init__(self)
        # create a normal distribution to model noise
        self._err = RVGs.Normal(loc=0, scale=err_sigma)
        # penalty
        self._penalty = penalty

    def get_obj_value(self, x, seed_index=0):
        """ returns one realization from x^2+noise """

        # create a random number generator
        rng = RVGs.RNG(seed=seed_index)

        accum_penalty = 0       # accumulated penalty

        # test the feasibility
        if x[1] < 1:
            accum_penalty += self._penalty * pow(x[1] - 1, 2)
            x[1] = 1

        return (x[0]+1)*(x[0]+1) + x[1]*x[1] + self._err.sample(rng) + accum_penalty
