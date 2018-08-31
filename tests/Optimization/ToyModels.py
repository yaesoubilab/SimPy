import SimPy.RandomVariantGenerators as RVGs
from SimPy.Optimization import SimModel

class Xto2(SimModel):
    def __init__(self, err_sigma):
        SimModel.__init__(self)
        self._rng = RVGs.RNG(seed=1)
        self._err = RVGs.Normal(loc=0, scale=err_sigma)

    def get_obj_value(self, x):
        return x*x + self._err.sample(self._rng)
