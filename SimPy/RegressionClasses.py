from numpy.polynomial import polynomial as P
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class _OneVarRegression:
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._coeffs = None

    def get_coeffs(self):
        return self._coeffs

    def get_predicted_y(self, x):
        pass

    def get_derivative(self, x):
        pass

    def get_roots(self):
        pass

    def plot_fit(self, x_range=None, fig_size=None):

        fig, ax = plt.subplots(figsize=fig_size)
        self.add_to_ax(ax=ax, x_range=x_range)
        fig.show()

    def add_to_ax(self, ax, x_range=None):

        if x_range:
            xs = np.linspace(x_range[0], x_range[-1], 50)
        else:
            xs = np.linspace(self._x[0], self._x[-1], 50)

        ys = self.get_predicted_y(xs)
        ax.scatter(self._x, self._y)
        ax.plot(xs, ys, color='red')


class PolyRegression(_OneVarRegression):
    # regression of form: f(x) = c0 + c1*x + c2*x^2 + c3*x^3 + ... + cn*x^n
    def __init__(self, x, y, degree=1):

        _OneVarRegression.__init__(self, x, y)
        self._coeffs = P.polyfit(x=x, y=y, deg=degree)

    def get_coeffs(self):
        return self._coeffs

    def get_predicted_y(self, x):

        f = 0
        for i, coeff in enumerate(self._coeffs):
            f += coeff * pow(x, i)
        return f

    def get_derivative(self, x):
        result = 0
        for i in range(len(self._coeffs) - 1):
            result += (i + 1) * self._coeffs[i + 1] * pow(x, i)

        return result

    def get_roots(self):
        return P.polyroots(self._coeffs)


class ExpRegression (_OneVarRegression):
    # regression of form f(x) = c0 + c1*exp(c2*x)

    def __init__(self, x, y, if_c0_zero=False, p0=None):

        _OneVarRegression.__init__(self, x, y)
        self._ifC0Zero = if_c0_zero
        if if_c0_zero:
            self._coeffs, cov = curve_fit(self.exp_func_c0_zero, x, y,
                                          p0=p0) #jac=self.Jac_c0_zero)
        else:
            self._coeffs, cov = curve_fit(self.exp_func, x, y,
                                          p0=p0) # jac=self.Jac)

    def get_predicted_y(self, x):
        if self._ifC0Zero:
            return self.exp_func_c0_zero(x, *self._coeffs)
        else:
            return self.exp_func(x, *self._coeffs)

    def get_derivative(self, x):
        if self._ifC0Zero:
            return self.der_exp_func_c0_zero(x, *self._coeffs)
        else:
            return self.der_exp_func(x, *self._coeffs)

    @staticmethod
    def exp_func(x, c0, c1, c2):
        return c0 + c1 * np.exp(c2 * x)

    @staticmethod
    def exp_func_c0_zero(x, c1, c2):
        return c1 * np.exp(c2 * x)

    @staticmethod
    def der_exp_func(x, c0, c1, c2):
        return c1 * c2 * np.exp(c2 * x)

    @staticmethod
    def der_exp_func_c0_zero(x, c1, c2):
        return c1 * c2 * np.exp(c2 * x)

    @staticmethod
    def Jac(x, c0, c1, c2):
        v = np.exp(c1*x)
        return np.array([np.ones(len(x)), v, c2*x*v]).transpose()

    @staticmethod
    def Jac_c0_zero(x, c1, c2):
        v = np.exp(c1*x)
        return np.array([v, c2*x*v]).transpose()


class PowerRegression (_OneVarRegression):
    # regression of form f(x) = c0 + c1*pow(x, c3)

    def __init__(self, x, y, if_c0_zero=False, p0=None):
        _OneVarRegression.__init__(self, x, y, )
        self._ifC0Zero = if_c0_zero
        if if_c0_zero:
            self._coeffs, cov = curve_fit(self.power_func_c0_zero, x, y, p0=None)
        else:
            self._coeffs, cov = curve_fit(self.power_func, x, y, p0=None)

    def get_coeffs(self):
        return self._coeffs

    def get_predicted_y(self, x):
        if self._ifC0Zero:
            return self.power_func_c0_zero(x, *self._coeffs)
        else:
            return self.power_func(x, *self._coeffs)



    @staticmethod
    def power_func(x, c0, c1, c2):
        return c0 + c1 * np.power(x, c2)


    @staticmethod
    def power_func_c0_zero(x, c1, c2):
        return c1 * np.power(x, c2)


# for additional information:
# http://markthegraph.blogspot.com/2015/05/using-python-statsmodels-for-ols-linear.html


class PolyRegFunction:
    # regression of form: f(x) = c0 + c1*x + c2*x^2 + c3*x^3 + ...

    def __init__(self, degree=1):
        """
        :param degree: degree of the polynomial function
        """
        if degree < 1:
            raise ValueError('Degree of the polynomial regression function should be greater than 0.')
        self.degree = degree

    def get_X(self, x):
        """
        :param x: (list or np.array) observed x values (x1, x2, x3, ..., xn)
        :return: (matrix) of X = [(1, x1'),
                                  (1, x2'),
                                  (1, x3'),
                                  ...,
                                  (1, xn')]
        """

        if self.degree == 1:
            # for degree 1, we only need to add constant vector 1 to x to get
            # [1, x1]
            # [1, x2] ...
            return sm.add_constant(x)
        else:
            col_x = [x]       # to build [x, x^2, x^3, ...]
            for i in range(2, self.degree+1):
                col_x.append(np.power(x, i))
            # turn [x, x^2, x^3, ...] to X
            X = np.column_stack(col_x)
            return sm.add_constant(X)


class SingleVarRegression:

    def __init__(self, x, y, degree=1):

        self.f = PolyRegFunction(degree)
        self.x = x
        self.X = self.f.get_X(x)
        self.y = y
        self.fitted = sm.OLS(self.y, self.X).fit()

    def get_predicted_y(self, x_pred):
        """ :returns predicted y values at the provided x values """

        X_pred = self.f.get_X(x_pred)
        return self.fitted.predict(X_pred)

    def get_predicted_y_CI(self, x_pred, alpha=0.05):
        """ :returns confidence interval of the predicted y at the provided x values """

        # http://www2.stat.duke.edu/~tjl13/s101/slides/unit6lec3H.pdf
        y_hat = self.fitted.predict(self.X)     # predicted y at X
        y_err = self.y - y_hat                  # residuals
        mean_x = self.x.mean()                  # mean of x
        n = len(self.X)                         # number of observations
        dof = n - self.fitted.df_model - 1      # degrees of freedom
        t = stats.t.ppf(1 - alpha/2, df=dof)      # t-statistics
        s_err = np.sum(np.power(y_err, 2))      # sum of squared error
        conf = t * np.sqrt((s_err / (n - 2)) * (1.0 / n + (np.power((x_pred - mean_x), 2) /
                                                           ((np.sum(np.power(x_pred, 2))) - n * (
                                                               np.power(mean_x, 2))))))
        y_pred = self.get_predicted_y(x_pred)
        upper = y_pred + abs(conf)
        lower = y_pred - abs(conf)

        return lower, upper

    def get_predicted_y_PI(self, x_pred, alpha=0.05):
        """ :returns prediction interval of the y at the provided x values """
        X_pred = self.f.get_X(x_pred)
        sdev, lower, upper = wls_prediction_std(self.fitted, exog=X_pred, alpha=alpha)
        return lower, upper

    def get_coeffs(self):
        """ :returns coefficients of the fitted model """

        return self.fitted.params

    def get_pvalues(self):
        """ :returns p-values of coefficients """
        return self.fitted.pvalues

    def get_derivative(self, x):
        """ :returns derivative of the polynomial function at x """

        coeffs = self.fitted.params
        result = 0
        for i in range(len(coeffs)-1):
            result += (i+1) * coeffs[i+1] * pow(x, i)

        return result

    def get_zero(self):
        """
        :return: x for which f(x) = 0
        """
        coeffs = np.fliplr([self.fitted.params])[0]
        if np.linalg.norm(coeffs) == 0:
            return np.zeros(len(coeffs))
        else:
            return np.roots(coeffs)



