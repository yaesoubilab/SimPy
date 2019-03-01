import SimPy.EconEvalClasses as Econ


print(Econ.pv(payment=10, discount_rate=0.05, discount_period=10))

print(Econ.pv(payment=10, discount_rate=0.05, discount_period=10, if_discount_continuously=True))
print(Econ.pv(payment=10, discount_rate=0.05/100, discount_period=10*100, if_discount_continuously=False))