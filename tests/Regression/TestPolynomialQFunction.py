import numpy as np
from SimPy.Regression import PolynomialQFunction

coeffs = np.array([-1, 1, 2, 3, -2, 4]) # coefficients of [1, x1, x2, x1^2, x1.x2, x1^2]
N = 1000
sigma = 1
l2_reg = 0.01
forgetting_factor = 0.95
degree = 2


def f(x):
    return coeffs @ np.array([1, x[0], x[1], pow(x[0], 2), x[0] * x[1], pow(x[1], 2)])


np.random.seed(seed=1)

# generate X
X = []
for n in range(N):
    # 2 random numbers over [-1, 1]
    x = 2*np.random.sample(2)-1
    X.append(x)

# find y's
y = [f(x) + np.random.normal(0, sigma) for x in X]

# fit a linear regression
q = PolynomialQFunction(degree=degree, l2_penalty=l2_reg)
for i in range(N):
    q.update(x=X[i], f=y[i])
print('\nQ-function: ')
print('Coeffs: ', q.get_coeffs(), 'vs.', coeffs)
print('f([1, -1]) = ', q.f(x=[1, -1]))
