from math import log, cos, pi

import numpy as np
from numpy import exp, pi, cos

from SimPy.RandomVariateGenerators import Beta as B
from SimPy.RandomVariateGenerators import Gamma as G
from SimPy.RandomVariateGenerators import Multinomial as Mult
from SimPy.RandomVariateGenerators import Uniform as U
from SimPy.RandomVariateGenerators import UniformDiscrete as UD


class _Parameter:
    # super class for parameters
    def __init__(self, id=None, name=None, if_time_dep=False):
        """
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        :param if_time_dep: (bool) if the value of this parameter changes by time
        """
        self.id = id
        self.name = name
        self.value = None
        self.ifTimeDep = if_time_dep

    def sample(self, rng=None, time=None):
        """
        :param rng: optional random number generator
        :param time: optimal time
        :return: a sample from this paramter
        """
        pass


class Constant(_Parameter):
    def __init__(self, value, id=None, name=None):
        """
        :param value: (float) constant value of this parameter
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        _Parameter.__init__(self, id=id, name=name)
        self.value = value

    def sample(self, rng=None, time=None):
        return self.value


class Uniform(_Parameter):
    def __init__(self, minimum=0, maximum=1, id=None, name=None):
        """
        :param minimum: (float) minimum value of a parameter with uniform distribution
        :param maximum: (float) maximum value of a parameter with uniform distribution
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name)
        self.par = U(scale=maximum-minimum, loc=minimum)

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class UniformDiscrete(_Parameter):
    def __init__(self, minimum=0, maximum=1, id=None, name=None):
        """
        :param minimum: (float) inclusive minimum value
        :param maximum: (float) inclusive maximum value
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name)
        self.par = UD(l=minimum, u=maximum)

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class Beta(_Parameter):
    def __init__(self, mean, st_dev, minimum=0, maximum=1, id=None, name=None):
        """
        :param mean: (float) mean value of a parameter with beta distribution
        :param st_dev: (float) st_dev value of a parameter with beta distribution
        :param minimum: (float) minimum value of a parameter with beta distribution
        :param maximum: (float) maximum value of a parameter with beta distribution
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        if not (minimum < mean < maximum):
            raise ValueError('Mean should be between minimum and maximum.')

        _Parameter.__init__(self, id=id, name=name)
        fit_results = B.fit_mm(mean=mean, st_dev=st_dev, minimum=minimum, maximum=maximum)
        self.par = B(a=fit_results['a'], b=fit_results['b'], loc=fit_results['loc'], scale=fit_results['scale'])

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class Gamma(_Parameter):
    def __init__(self, mean, st_dev, id=None, name=None):
        """
        :param mean: (float) mean value of a parameter with beta distribution
        :param st_dev: (float) st_dev value of a parameter with beta distribution
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        _Parameter.__init__(self, id=id, name=name)
        fit_results = G.fit_mm(mean=mean, st_dev=st_dev)
        self.par = G(a=fit_results['a'], scale=fit_results['scale'])

    def sample(self, rng=None, time=None):
        self.value = self.par.sample(rng=rng)
        return self.value


class Equal(_Parameter):
    # value = value of another parameter

    def __init__(self, par, id=None, name=None):
        """
        :param par: (Parameter) another parameter to set this parameter equal to
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=par.ifTimeDep)
        self.par = par

    def sample(self, rng=None, time=None):
        self.value = self.par.value
        return self.value


class Multinomial(_Parameter):
    def __init__(self, par_n, p_values, id=None, name=None):
        """
        :param par_n: (Parameter) number of trials
        :param p_values: (array) probabilities of success for each category
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        if not (0.9999999 <= sum(p_values) <= 1.0000001):
            raise ValueError("Sum of p_values should be 1 (not {}) for parameter '{}'.".format(sum(p_values), name))

        _Parameter.__init__(self, id=id, name=name)
        self.parN = par_n
        self.pVals = p_values

    def sample(self, rng=None, time=None):

        self.value = Mult(N=self.parN.value, pvals=self.pVals).sample(rng=rng)
        return self.value


class AMultinomialOutcome(_Parameter):
    # a parameter that is defined on one outcome of a multinomial parameter
    def __init__(self, par_multinomial, outcome_index, id=None, name=None):
        """
        :param par_multinomial: (Parameter) a multinomial parameter
        :param outcome_index: (int) index of the outcome of interest
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name)
        self.multinomial = par_multinomial
        self.i = outcome_index

    def sample(self, rng=None, time=None):
        self.value = self.multinomial.value[self.i]
        return self.value


class Inverse(_Parameter):
    # value = 1 / value of another parameter

    def __init__(self, par, id=None, name=None):
        """
        :param par: (Parameter) another parameter to use to calculate the inverse
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=par.ifTimeDep)
        self.par = par

    def sample(self, rng=None, time=None):
        self.value = 1/self.par.value
        return self.value


class OneMinus(_Parameter):
    # value = 1 - value of another parameter

    def __init__(self, par, id=None, name=None):
        """
        :param par: (Parameter) another parameter to use to calculate 1 - p
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=par.ifTimeDep)
        self.par = par

    def sample(self, rng=None, time=None):
        self.value = 1-self.par.value
        return self.value


class TenToPower(_Parameter):
    # 10^(value of another parameter)

    def __init__(self, par, id=None, name=None):
        """
        :param par: (Parameter) another parameter to use to calculate 10^p
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=par.ifTimeDep)
        self.par = par

    def sample(self, rng=None, time=None):
        self.value = pow(10, self.par.value)
        return self.value


class Logit(_Parameter):
    # p/(1-p)

    def __init__(self, par, id=None, name=None):
        """
        :param par: (Parameter) another parameter to use to calculate p(1 - p)
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=par.ifTimeDep)
        self.par = par

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

        _Parameter.__init__(self, id=id, name=name, if_time_dep=par_probability.ifTimeDep)
        self.parProb = par_probability
        self.deltaTInv = 1/delta_t

    def sample(self, rng=None, time=None):
        self.value = -log(1-self.parProb.value) * self.deltaTInv
        return self.value


class Division(_Parameter):
    # p1 / p2
    def __init__(self, par_numerator, par_denominator, id=None, name=None):
        """
        :param par_numerator: (Parameter) numerator parameter
        :param par_denominator: (Parameter) denominator parameter
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=(par_numerator.ifTimeDep or par_denominator.ifTimeDep))
        self.numerator = par_numerator
        self.denominator = par_denominator

    def sample(self, rng=None, time=None):
        self.value = self.numerator.value/self.denominator.value
        return self.value


class LinearCombination(_Parameter):
    # linear combination of multiple parameters

    def __init__(self, parameters, coefficients=None, id=None, name=None):
        """
        :param parameters: (list of Parameters)
        :param coefficients: (list of floats) list of coefficients
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        # if this is a time-dependent parameter
        if_time_dep = False
        for p in parameters:
            if p.ifTimeDep:
                if_time_dep = True
                break

        _Parameter.__init__(self, id=id, name=name, if_time_dep=if_time_dep)
        self.parameters = parameters
        if coefficients is not None:
            self.coefficients = coefficients
        else:
            self.coefficients = [1] * len(parameters)

    def sample(self, rng=None, time=None):
        self.value = 0
        for i, p in enumerate(self.parameters):
            self.value += p.value * self.coefficients[i]

        return self.value


class OneMinusSum(_Parameter):
    # 1 - sum of of multiple parameters

    def __init__(self, parameters, id=None, name=None):
        """
        :param parameters: (list of Parameters)
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        # if this is a time-dependent parameter
        if_time_dep = False
        for p in parameters:
            if p.ifTimeDep:
                if_time_dep = True
                break

        _Parameter.__init__(self, id=id, name=name, if_time_dep=if_time_dep)
        self.params = parameters

    def sample(self, rng=None, time=None):

        self.value = 1
        for p in self.params:
            self.value -= p.value
        return self.value


class Product(_Parameter):
    # product of multiple parameters

    def __init__(self, parameters, id=None, name=None):
        """
        :param parameters: (list of Parameters)
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        # if this is a time-dependent parameter
        if_time_dep = False
        for p in parameters:
            if p.ifTimeDep:
                if_time_dep = True
                break

        _Parameter.__init__(self, id=id, name=name, if_time_dep=if_time_dep)
        self.parameters = parameters

    def sample(self, rng=None, time=None):
        self.value = 1
        for p in self.parameters:
            self.value *= p.value

        return self.value


class Surge(_Parameter):
    # f(t) = base.value * ( 1 + percentChange(t))
    # percentChange(t) = A * (1 - cos(2*pi*(t1-t)/(t1-t0)) / 2
    # A is maximum % change
    def __init__(self, par_base=1, par_max_percent_change=1, par_t0=0, par_t1=1, id=None, name=None):
        """
        :param par_base: (Parameter or float) value to use as base
        :param par_max_percent_change: (Parameter or float) maximum % change in base value
            (should be + for increase and - for decrease)
        :param par_t0: (Parameter or float) f(t) = 0 for t < t0
        :param par_t1: (Parameter or float) f(t) = 0 for t > t1
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """

        if not isinstance(par_base, _Parameter):
            par_base = Constant(value=par_base)
        if not isinstance(par_max_percent_change, _Parameter):
            par_max_percent_change = Constant(value=par_max_percent_change)
        if not isinstance(par_t0, _Parameter):
            par_t0 = Constant(value=par_t0)
        if not isinstance(par_t1, _Parameter):
            par_t1 = Constant(value=par_t1)

        _Parameter.__init__(self=self, id=id, name=name,
                            if_time_dep=True)

        self.base = par_base
        self.maxPercChange = par_max_percent_change
        self.t0 = par_t0
        self.t1 = par_t1
        self.twoPi = 2*pi

    def sample(self, rng=None, time=None):

        if time < self.t0.value or time > self.t1.value:
            self.value = self.base.value
        else:
            x = self.twoPi * (time-self.t0.value)/(self.t1.value - self.t0.value)
            percent_change = self.maxPercChange.value * (1 - cos(x)) / 2
            self.value = self.base.value * (1 + percent_change)
        return self.value


class TimeDependentSigmoid(_Parameter):
    # f(t) = min + (max-min) * 1 / (1 + exp(-b * (t - t_middle - t_min)) if t > t_min
    # returns min for t = -inf and max for t = inf if b >= 0

    def __init__(self, par_b, par_t_min=None, par_t_middle=None, par_min=None, par_max=None, id=None, name=None):
        """
        :param par_b: (Parameter) of b
        :param par_t_min: (Parameter) of t_min
        :param par_t_middle: (Parameter) of t_middle
        :param par_min: (Parameter) of min (if not provided, Constant(0) is used)
        :param par_max: (Parameter) of max (if not provided, Constant(1) is used)
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=True)

        self.parB = par_b
        self.parTMin = par_t_min if par_t_min is not None else Constant(value=0)
        self.parTMid = par_t_middle if par_t_middle is not None else Constant(value=0)
        self.parMin = par_min if par_min is not None else Constant(value=0)
        self.parMax = par_max if par_max is not None else Constant(value=1)

    def sample(self, rng=None, time=None):

        if time < self.parTMin.value:
            self.value = 0
        else:
            dt = time - self.parTMid.value - self.parTMin.value
            logistic = 1 / (1 + exp(-self.parB.value * dt))
            self.value = self.parMin.value + (self.parMax.value - self.parMin.value) * logistic

        return self.value


class TimeDependentCosine(_Parameter):
    # f(t) = min + (max-min) * Cos (2 * pi * (t - phase)/scale)

    def __init__(self, par_phase=None, par_scale=None, par_min=None, par_max=None, id=None, name=None):
        """
        :param par_phase: (Parameter) of phase
        :param par_scale: (Parameter) of scale
        :param par_min: (Parameter) of min (if not provided, Constant(0) is used)
        :param par_max: (Parameter) of max (if not provided, Constant(1) is used)
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=True)

        self.parPhase = par_phase
        self.parScale = par_scale
        self.parMin = par_min if par_min is not None else Constant(value=0)
        self.parMax = par_max if par_max is not None else Constant(value=1)

    def sample(self, rng=None, time=None):

        arg = (time - self.parPhase.value)/self.parScale.value
        cosine = cos(2*pi*arg)
        self.value = self.parMin.value + (self.parMax.value - self.parMin.value) * (cosine + 1) / 2

        return self.value


class TimeDependentStepWise(_Parameter):
    # f(t) = 0  for       t < t0
    #      = v0 for t0 <= t < t1
    #      = v1 for t1 <= t < t2
    #      = v2 for t2 <= t

    def __init__(self, ts, vs, id=None, name=None):
        """
        :param ts: (list) of time break points
        :param vs: (list) of values
        :param id: (int) id of a parameter
        :param name: (string) name of a parameter
        """
        _Parameter.__init__(self, id=id, name=name, if_time_dep=True)

        assert len(ts) == len(vs), 'There should be an equal number of time breakpoints (ts) and function values (vs).'

        self.ts = ts
        self.vs = vs

    def sample(self, rng=None, time=None):

        raise ValueError('Needs to be debugged.')

        self.value = 0

        if time < self.ts[0]:
            self.value = 0
        else:
            i = 0
            while True:
                if time < self.ts[i]:
                    break
                else:
                    i += 1

            self.value = self.vs[i]

        return self.value


class MatrixOfParams(_Parameter):
    def __init__(self, matrix_of_params_or_values, id=None, name=None):
        """
        :param matrix_of_params_or_values: (list of list) of numbers or Parameters
        :param id:
        :param name:
        """

        self.matrixOfParams = []
        for row in matrix_of_params_or_values:
            params = []
            for v in row:
                if isinstance(v, _Parameter):
                    params.append(v)
                else:
                    params.append(Constant(value=v))
            self.matrixOfParams.append(params)

        # find if time-dependant
        if_time_dep = False
        for row in self.matrixOfParams:
            for v in row:
                if v.ifTimeDep:
                    if_time_dep = True
                    break
            if if_time_dep:
                break

        _Parameter.__init__(self, id=id, name=name, if_time_dep=if_time_dep)

        if not self.ifTimeDep:
            self.sample()

    def sample(self, rng=None, time=None):
        """
        :return: (np.array)
        """

        self.value = []
        for params in self.matrixOfParams:
            values = []
            for par in params:
                values.append(par.value)
            self.value.append(values)

        self.value = np.array(self.value)
        return self.value


class ValuesOfParams(_Parameter):
    """ returns the values of a list of parameters """

    def __init__(self, parameters, id=None, name=None):
        """
        :param parameters: (list) of Parameters
        :param id:
        :param name:
        """

        self.parameters = parameters
        # find if time-dependant
        if_time_dep = False
        for p in self.parameters:
            if p.ifTimeDep:
                if_time_dep = True
                break
        _Parameter.__init__(self, id=id, name=name, if_time_dep=if_time_dep)

    def sample(self, rng=None, time=None):

        self.value = []
        for param in self.parameters:
            self.value.append(param.value)
        return self.value

