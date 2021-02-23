from numpy.polynomial import polynomial as P
import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# ------- Single variable regression models ----------
class _OneVarRegression:

    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._coeffs = None

    def get_coeffs(self):
        return self._coeffs

    def get_predicted_y(self, x):
        raise NotImplementedError

    def get_derivative(self, x):
        raise NotImplementedError

    def get_roots(self):
        raise NotImplementedError

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
            self._coeffs, cov = curve_fit(self.power_func_c0_zero, x, y, p0=p0)
        else:
            self._coeffs, cov = curve_fit(self.power_func, x, y, p0=p0)

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


# ------- Single variable polynomial model with inference ----------
class SingleVarPolyRegWithInference:

    # for additional information:
    # http://markthegraph.blogspot.com/2015/05/using-python-statsmodels-for-ols-linear.html

    def __init__(self, x, y, degree=1):

        self.degree = degree
        self.x = x
        self.X = self.get_X(x)
        self.y = y
        self.fitted = sm.OLS(self.y, self.X).fit()

    def get_predicted_y(self, x_pred):
        """ :returns predicted y values at the provided x values """

        X_pred = self.get_X(x_pred)
        return self.fitted.predict(X_pred)

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
        X_pred = self.get_X(x_pred)
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


# ------- Multi-variable linear regression ----------
class LinearRegression:
    
    def __init__(self, l2_penalty=0):
        """
        :param l2_penalty: (float) l2 regularization penalty
        """

        self.l2Penalty = l2_penalty
        self._coeffs = None

    def fit(self, X, y, forgetting_factor=1):

        # W
        N = len(y)
        if forgetting_factor < 1:
            w = []
            for i in range(N):
                w.append(pow(forgetting_factor, N - i - 1))
            W = np.diag(w)
        else:
            W = np.diag([1]*N)

        # XT.W.X
        XTX = np.transpose(X) @ W @ X
        # XT.W.y
        XTy = np.transpose(X) @ W @ y

        if self.l2Penalty > 0:
            self._add_l2(XTX)

        # solve to estiamte the coefficients
        self._coeffs = np.linalg.solve(XTX, XTy)

    def get_coeffs(self):
        return self._coeffs

    def set_coeffs(self, values):
        self._coeffs = values

    def get_y(self, x):
        return x @ self._coeffs

    def _add_l2(self, XTX):
        I = np.identity(XTX.shape[0])
        XTX += I * self.l2Penalty


class RecursiveLinearReg(LinearRegression):

    def __init__(self, l2_penalty=0):
        """
        :param l2_penalty: (float) l2 regularization penalty
        """

        LinearRegression.__init__(self, l2_penalty=l2_penalty)

        self.itr = 0
        self._X = None
        self._y = None
        self._B = None
        self._H = None
        self._w = None

    def update(self, x, y, forgetting_factor=1):

        self.itr += 1

        if self.itr < len(x):
            if self._X is None:
                self._X = np.array(x)
            else:
                self._X = np.vstack((self._X, x))
            if self._y is None:
                self._y = np.array(y)
            else:
                self._y = np.vstack((self._y, y))
            if self._w is None:
                self._w = np.array(1.0)
            else:
                self._w *= forgetting_factor
                self._w = np.append(self._w, 1.0)

        elif self.itr == len(x):
            self._X = np.vstack((self._X, x))
            self._y = np.vstack((self._y, y))
            self._w *= forgetting_factor
            self._w = np.append(self._w, 1.0)
            W = np.diag(self._w)

            # XTX
            XTX = np.transpose(self._X) @ W @ self._X

            # add L2 regularization
            if self.l2Penalty > 0:
                self._add_l2(XTX)

            # B = (XT.X)-1
            self._B = np.linalg.inv(XTX)
            # theta = B.XT.y
            self._coeffs = np.transpose(self._B @ np.transpose(self._X) @ W @ self._y)[0]
        else:
            # turn x into a column vector
            x = np.transpose(np.asmatrix(x))
            # gamma = lambda + xT.B.x
            gamma = float(forgetting_factor + np.transpose(x) @ self._B @ x)
            # epsilon = y - thetaT*x
            epsilon = float(y - self._coeffs @ x)
            # theta = theta + B.x.epsilon/gamma
            self._coeffs += np.transpose(self._B @ x * epsilon / gamma).A1
            # B = (B-B.x.xT.B/gamma)/lambda
            d = self._B @ x @ np.transpose(x) @ self._B
            self._B -= d / gamma

            self._B /= forgetting_factor


# ------- Q-functions ----------
class _QFunction:

    def __init__(self, name=None):
        self.name = name

    def update(self, x, f, forgetting_factor=1):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError


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
        return self.reg.get_coeffs()

    def set_coeffs(self, values):
        self.reg.set_coeffs(values=values)
