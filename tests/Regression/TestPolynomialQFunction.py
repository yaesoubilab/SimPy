import numpy as np

from SimPy.Regression import PolynomialQFunction

N = 1000
sigma = 0
l2_reg = 0.0
forgetting_factor = 1 + 0 * 0.95
degree = 2

# coefficients of [1, x1, x2, x1^2, x1.x2, x1^2] if the categorical variable i 0
COEFFS_C0 = np.array([-1, 1, 2, 3, -2, 4])
# coefficients of [1, x1, x2, x1^2, x1.x2, x1^2] if the categorical variable is 1
COEFFS_C1 = np.array([1, -1, 3, -2, 3, 1])


def f(x):
    """
    :param x: (list) continuous variables
    :return: f(x)
    """
    return COEFFS_C0 @ np.array([1, x[0], x[1], pow(x[0], 2), x[0] * x[1], pow(x[1], 2)])


def g(x, i):
    """
    :param x: (list) continuous variables
    :param i: categorical variable
    :return: g(x, i)
    """
    if i == 0:
        return COEFFS_C0 @ np.array([1, x[0], x[1], pow(x[0], 2), x[0] * x[1], pow(x[1], 2)])
    elif i == 1:
        return COEFFS_C1 @ np.array([1, x[0], x[1], pow(x[0], 2), x[0] * x[1], pow(x[1], 2)])
    else:
        raise ValueError('Invalid value of categorical variable')


np.random.seed(seed=1)

# generate X (to store observations on the continuous variable)
# and I (to store observations ont the categorical variable)
X = []
I = []
for n in range(N):

    # 2 random numbers over [-1, 1] for (x1, x2)
    x = 2*np.random.sample(2)-1

    # 0 or 1 for the categorical variable
    i = 0 if np.random.random_sample() < 0.5 else 1

    X.append(x)
    I.append(i)

# find f(x)
fs = [f(x) + np.random.normal(0, sigma) for x in X]
gs = [g(x, i) + np.random.normal(0, sigma) for x, i in zip(X, I)]

# fit a Q-function with continuous variables only
q_cont = PolynomialQFunction(degree=degree, l2_penalty=l2_reg)
for i in range(N):
    q_cont.update(f=fs[i], continuous_features=X[i])
print('\nQ-function with continuous variables: ')
print('Coeffs: ', q_cont.get_coeffs(), 'vs.', COEFFS_C0)
print('f([1, -1]) should be 7 = ', q_cont.f(continuous_features=[1, -1]))

# fit a Q-function with continuous and categorical variables
q_cont_cat = PolynomialQFunction(degree=degree, l2_penalty=l2_reg)
for i in range(N):
    q_cont_cat.update(f=gs[i], continuous_features=X[i], categorical_features=I[i])
print('\nQ-function with continuous and categorical variables: ')
print('Coeffs: ', q_cont.get_coeffs(), 'vs.', COEFFS_C0 + COEFFS_C1)
print('f([1, -1, 0] should be ) = ', q_cont.f(continuous_features=[1, -1], categorical_features=0))
