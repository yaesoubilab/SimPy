from SimPy.Support.EconEvalSupport import *


w = find_intersecting_wtp(w0=0,
                          u_new=inmb_u(d_cost=5000, d_effect=0),
                          u_base=inmb_u(d_cost=0, d_effect=0))
print(w)

