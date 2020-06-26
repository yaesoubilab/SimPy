
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


class Division(_Parameter):
    def __init__(self, numerator, denominator, id=0):
        _Parameter.__init__(self, id=id)

        self.numerator = numerator
        self.denominator = denominator
        self.value = numerator.value/denominator.value

    def sample(self, rng=None, time=None):
        self.value = self.numerator.value/self.denominator.value
