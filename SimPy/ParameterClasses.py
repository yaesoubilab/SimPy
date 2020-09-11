
class _Parameter:
    
    def __init__(self, id):
        self.id = id
        self.value = None

    def sample(self, rng=None, time=None):
        pass


class Constant(_Parameter):
    def __init__(self, value, id=0):

        _Parameter.__init__(self, id=id)
        self.value = value

    def sample(self, rng=None, time=None):
        pass


class Inverse(_Parameter):
    def __init__(self, par, id=0):

        _Parameter.__init__(self, id=id)
        self.par = par
        self.value = 1 / self.par.value

    def sample(self, rng=None, time=None):
        self.value = 1/self.par.value


class Division(_Parameter):
    def __init__(self, par_numerator, par_denominator, id=0):
        _Parameter.__init__(self, id=id)

        self.numerator = par_numerator
        self.denominator = par_denominator
        self.value = self.numerator.value / self.denominator.value

    def sample(self, rng=None, time=None):
        self.value = self.numerator.value/self.denominator.value
