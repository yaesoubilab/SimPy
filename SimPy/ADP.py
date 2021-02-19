from sklearn.preprocessing import PolynomialFeatures
from SimPy.Regression import RecursiveLinearReg


class _QFunction:

    def __init__(self, name=None):

        self.name = name

    def update(self, x, f, forgetting_factor=1):
        pass

    def f(self, x):
        pass


class PolynomialQFunction(_QFunction):

    def __init__(self, name=None, degree=2, l2_penalty=0):

        _QFunction.__init__(self, name=name)

        self.poly = PolynomialFeatures(degree=degree)
        self.reg = RecursiveLinearReg(l2_penalty=l2_penalty)

    def update(self, x, f, forgetting_factor=1):

        self.reg.update(x=self.poly.fit_transform(X=[x])[0],
                        y=f, forgetting_factor=forgetting_factor)

    def f(self, x):

        return self.reg.get_y(x=self.poly.fit_transform(X=[x])[0])

    def get_coeffs(self):
        return self.reg.coeffs
