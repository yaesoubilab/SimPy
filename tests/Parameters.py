from SimPy.Parameters import Surge, Drop
import numpy as np

surge = Surge(par_max=2, par_t0=0, par_t1=2)
drop = Drop(par_min=-2, par_t0=0, par_t1=2)

ts = np.linspace(0.0, 2.0, 20)

surge_fs = [surge.sample(time=t) for t in ts]
drop_fs = [drop.sample(time=t) for t in ts]

print(ts)
print(surge_fs)
print(drop_fs)
