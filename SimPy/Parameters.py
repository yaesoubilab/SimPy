from math import log
from SimPy.RandomVariateGenerators import Beta as B
from SimPy.RandomVariateGenerators import Uniform as U
import numpy as np


class _Parameter:
    
    def __init__(self, id=None, name=None):
        self.id = id
        self.name = name
        self.value = None

    def sample(self, rng=None, time=None):
        pass


class Constant(_Parameter):
    def __init__(self, value, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.value = value

    def sample(self, rng=None, time=None):
        return self.value


class Uniform(_Parameter):
    def __init__(self, minimum=0, maximum=1, id=None, name=None):
        _Parameter.__init__(self, id=id, name=name)
        self.par = U(scale=maximum-minimum, loc=minimum)
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class Beta(_Parameter):
    def __init__(self, mean, st_dev, minimum=0, maximum=1, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        fit_results = B.fit_mm(mean=mean, st_dev=st_dev, minimum=minimum, maximum=maximum)
        self.par = B(a=fit_results['a'], b=fit_results['b'], loc=fit_results['loc'], scale=fit_results['scale'])
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class Inverse(_Parameter):
    def __init__(self, par, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.par = par
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = 1/self.par.value
        return self.value


class OneMinus(_Parameter):
    def __init__(self, par, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.par = par
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = 1-self.par.value
        return self.value


class TenToPower(_Parameter):
    # 10^p
    def __init__(self, par, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.par = par
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = pow(10, self.par.value)
        return self.value


class Logit(_Parameter):
    # p/(1-p)
    def __init__(self, par, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.par = par
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = self.par.value/(1-self.par.value)
        return self.value


class RateToOccur(_Parameter):
    """ determines rate of an event such that it occurs with certain probability during a certain period """
    def __init__(self, par_probability, delta_t, id=None, name=None):
        """
        :param par_probability: parameter for the probability that the event occurs during deltaT
        :param delta_t: time period over which the event should occur with the specified probability
        """

        _Parameter.__init__(self, id=id, name=name)
        self.parProb = par_probability
        self.deltaTInv = 1/delta_t
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = -log(1-self.parProb.value) * self.deltaTInv
        return self.value


class Division(_Parameter):
    def __init__(self, par_numerator, par_denominator, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.numerator = par_numerator
        self.denominator = par_denominator
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = self.numerator.value/self.denominator.value
        return self.value


class LinearCombination(_Parameter):
    def __init__(self, parameters, coefficients=None, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.parameters = parameters
        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self.coefficients = [1] * len(parameters)
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = 0
        for i, p in enumerate(self.parameters):
            self.value += p.value * self.coefficients[i]

        return self.value


class Product(_Parameter):
    def __init__(self, parameters, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.parameters = parameters
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = 1
        for p in self.parameters:
            self.value *= p.value

        return self.value


class MatrixOfConstantParams(_Parameter):
    def __init__(self, matrix_of_values, id=None, name=None):
        """
        :param matrix_of_values: (list of list) of numbers
        :param id:
        :param name:
        """

        _Parameter.__init__(self, id=id, name=name)

        self.matrixOfParams = []
        for row in matrix_of_values:
            params = []
            for v in row:
                params.append(Constant(value=v))
            self.matrixOfParams.append(params)

        self.sample()

    def sample(self, rng=None, time=None):

        self.value = []
        for params in self.matrixOfParams:
            values = []
            for par in params:
                values.append(par.value)
            self.value.append(values)

        self.value = np.array(self.value)
        return self.value
