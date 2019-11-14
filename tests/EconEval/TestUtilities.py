import SimPy.Support.EconEvalSupport as S

u = S.inmb2_u(d_effect=20, d_cost=1000)
print(u(w_gain=100, w_loss=200))

u = S.inmb2_u(d_effect=-1, d_cost=1000)
print(u(w_gain=100, w_loss=200))

