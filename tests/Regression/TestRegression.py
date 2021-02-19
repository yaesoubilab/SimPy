import numpy as np
from SimPy.Regression import LinearRegression, RecursiveLinearReg

coeffs = [1, 2, 3]  # y = 1 + 2x1 + 3x2
N = 1000
sigma = 1
l2_reg = 0.01
forgetting_factor = 0.95

np.random.seed(seed=1)

# generate X
X = []
for n in range(N):
    # 3 random numbers over [-1, 1]
    x = 2*np.random.sample(3)-1
    X.append(x)

# find y's
y = np.dot(X, np.array(coeffs)) + np.random.normal(0, sigma)

# fit a linear regression
lr = LinearRegression()
lr.fit(X=X, y=y)
print('Regression: ')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]) = ', lr.get_y(x=[1, -1, 2]))

# fit a linear regression with L2 regularization
lr = LinearRegression(l2_penalty=l2_reg)
lr.fit(X=X, y=y)
print('\nRegression (with l2-regularization):')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]) = ', lr.get_y(x=[1, -1, 2]))

# fit a linear regression with forgetting factor
lr = LinearRegression()
lr.fit(X=X, y=y, forgetting_factor=forgetting_factor)
print('\nRegression (with forgetting factor):')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]) = ', lr.get_y(x=[1, -1, 2]))

# recursive a linear regression
lr = RecursiveLinearReg()
for i in range(N):
    lr.update(x=X[i], y=y[i])
print('\nRecursive regression: ')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]', lr.get_y(x=[1, -1, 2]))

lr = RecursiveLinearReg(l2_penalty=l2_reg)
for i in range(N):
    lr.update(x=X[i], y=y[i])
print('\nRecursive regression (with l2-regularization): ')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]', lr.get_y(x=[1, -1, 2]))

lr = RecursiveLinearReg()
for i in range(N):
    lr.update(x=X[i], y=y[i], forgetting_factor=forgetting_factor)
print('\nRecursive regression (with forgetting factor): ')
print('Coeffs: ', lr.get_coeffs(), 'vs.', coeffs)
print('y([1, -1, 2]', lr.get_y(x=[1, -1, 2]))
