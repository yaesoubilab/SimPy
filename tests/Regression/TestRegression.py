import SimPy.RegressionClasses as Reg
import numpy as np
import matplotlib.pyplot as plt

# generate x values (observations)
x = np.random.randn(100)
# generate y values (assuming y = x^2 -2 + error)
y = x*x + np.random.randn(100) - 2

# create the scatter plot of (x, y) points
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, alpha=0.5, color='orchid')
fig.suptitle('Example Scatter Plot')
fig.tight_layout(pad=2)
ax.grid(True)

# create the regression model
single_var_poly_reg = Reg.SingleVarRegression(x, y, degree=2)

# print the coefficients of the fitted model
print('Estimated coefficients:', single_var_poly_reg.get_coeffs())

# print derivative
print('Derivative at x=1:', single_var_poly_reg.get_derivative(x=1))

# print zeros
print('x for which f(x) = 0:', single_var_poly_reg.get_zero())

# make prediction over the range [x_min, x_max]
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred = single_var_poly_reg.get_predicted_y(x_pred)

# plot the predicted values
ax.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2)

# add the confidence region of predictions
lower, upper = single_var_poly_reg.get_predicted_y_CI(x_pred)
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4)

# add the prediction intervals of predictions
lower, upper = single_var_poly_reg.get_predicted_y_PI(x_pred)
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1)

plt.show()
