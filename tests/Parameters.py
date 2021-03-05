import numpy as np

from SimPy.Parameters import Surge

# 200% increase from the base value which has a default value of 1
surge = Surge(par_max_percent_change=2, par_t0=0, par_t1=2)
# 100% decrease from the base value which has a default value of 0
drop = Surge(par_max_percent_change=-1, par_t0=0, par_t1=2)

ts = np.linspace(0.0, 2.0, 21)

surge_fs = [surge.sample(time=t) for t in ts]
drop_fs = [drop.sample(time=t) for t in ts]

print(ts)
print(surge_fs)
print(drop_fs)
