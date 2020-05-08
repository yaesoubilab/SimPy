from numpy.polynomial import polynomial as P
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import curve_fit


class PolyRegression:
    # regression of form: f(x) = c0 + c1*x + c2*x^2 + c3*x^3 + ... + cn*x^n
    def __init__(self, x, y, degree=1):

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


class ExpRegression:
    # regression of form f(x) = c0 + c1*exp(c2*x)

    def __init__(self, x, y, if_zero_at_limit=False):

        self._ifZeroAtLimit = if_zero_at_limit
        if if_zero_at_limit:
            self._coeffs, cov = curve_fit(self.exp_func_zero_at_limit, x, y)
        else:
            self._coeffs, cov = curve_fit(self.exp_func, x, y)

    def get_coeffs(self):
        return self._coeffs

    def get_predicted_y(self, x):
        if self._ifZeroAtLimit:
            return self.exp_func_zero_at_limit(x, *self._coeffs)
        else:
            return self.exp_func(x, *self._coeffs)

    @staticmethod
    def exp_func(x, c0, c1, c2):
        return c0 + c1 * np.exp(c2 * x)
    @staticmethod
    def exp_func_zero_at_limit(x, c1, c2):
        return c1 * np.exp(c2 * x)
