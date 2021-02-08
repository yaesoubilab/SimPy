from SimPy.Parameters import Surge
import numpy as np

shock = Surge(par_max=2, par_t0=0, par_t1=2)

ts = np.linspace(0.0, 2.0, 20)

fs = [shock.sample(time=t) for t in ts]

print(ts)
print(fs)
