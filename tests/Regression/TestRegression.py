import SimPy.RegressionClasses as Reg
import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(100)
y = x*x + np.random.randn(100)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, alpha=0.5, color='orchid')
fig.suptitle('Example Scatter Plot')
fig.tight_layout(pad=2)
ax.grid(True)

# regression
single_var_poly_reg = Reg.SingleVarRegression(x, y, degree=3)

x_pred = np.linspace(x.min(), x.max(), 50)
y_pred = single_var_poly_reg.get_predicted_y(x_pred)

ax.plot(x_pred, y_pred, '-', color='darkorchid', linewidth=2)
lower, upper = single_var_poly_reg.get_predicted_y_CI(x_pred)

ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.4)
#
lower, upper = single_var_poly_reg.get_predicted_y_PI(x_pred)
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1)

plt.show()