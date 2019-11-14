from SimPy.Support.EconEvalSupport import *


w = find_intersecting_wtp(w0=0,
                          u_new=inmb_u(d_cost=5000, d_effect=0),
                          u_base=inmb_u(d_cost=0, d_effect=0))
print(w)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

# Parallelogram

x = 0.5
width = 0.1
mean = 0.5
error = 0.1
xy = [
    [x, mean-error],
    [x-width, mean],
    [x, mean+error],
    [x+width, mean]
]

x = [0.3,0.6,.7,.4]
y = [0.4,0.4,0.6,0.6]
ax.add_patch(patches.Polygon(xy=xy, fill=False))


plt.show()