
class _Parameter:
    
    def __init__(self, id):
        self.id = id
        self.value = None

    def sample(self, rnd=None):
        pass


class Constant(_Parameter):
    def __init__(self, id, value):

        _Parameter.__init__(self, id=id)
        self.value = value

    def sample(self, rnd=None):
        pass
