
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
        pass


class BetaWithConfidenceInterval(_Parameter):
    def __init__(self, mean, confidence_interval, confidence_level=0.05, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.parameters = 0

    def sample(self, rng=None, time=None):
        pass


class Inverse(_Parameter):
    def __init__(self, par, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.par = par
        self.value = 1 / self.par.value

    def sample(self, rng=None, time=None):
        self.value = 1/self.par.value


class Division(_Parameter):
    def __init__(self, par_numerator, par_denominator, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.numerator = par_numerator
        self.denominator = par_denominator
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = self.numerator.value/self.denominator.value


class Product(_Parameter):
    def __init__(self, parameters, id=None, name=None):

        _Parameter.__init__(self, id=id, name=name)
        self.parameters = parameters
        self.sample()

    def sample(self, rng=None, time=None):
        self.value = 1
        for p in self.parameters:
            self.value *= p.value
