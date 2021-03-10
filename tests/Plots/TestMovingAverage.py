import matplotlib.pyplot as plt
import numpy as np

import SimPy.Plots.FigSupport as Fig

data = [1, None, 3, 4, 5, 6, 7, None, 9]
print(Fig.get_moving_average(data=data, window=3))


data = np.random.randn(100)

fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
ax.plot(range(100), data, alpha=0.5)
ax.plot(range(100), Fig.get_moving_average(data, window=10), alpha=1)

plt.show()
