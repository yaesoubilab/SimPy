import numpy as np
from SimPy.Regression import LinearRegression

coeffs = [1, 2, 3]  # y = 1 + 2x1 + 3x2
N = 100
sigma = 0

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
print('Coeffs: ', lr.coeffs, 'vs.', coeffs)

# fit a linear regression with L2 regularization
lr = LinearRegression(l2_penalty=0.01)
lr.fit(X=X, y=y)
print('Coeffs (with l2-regularization): ', lr.coeffs, 'vs.', coeffs)

