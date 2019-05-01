import SimPy.EconEval as Econ


print('Present value of $10 collected in year 20 at discount rate 5%:')
print(Econ.pv_single_payment(payment=10, discount_rate=0.05, discount_period=20))

print('\nPresent value of $10 collected in year 20 at discount rate 5% (discounted continuously):')
print('These 2 numbers should be almost the same:')
print(Econ.pv_single_payment(payment=10, discount_rate=0.05,
                             discount_period=20, discount_continuously=True))
print(Econ.pv_single_payment(payment=10, discount_rate=0.05 / 100,
                             discount_period=20 * 100, discount_continuously=False))

print('\nPresent value of a continuous payment of $10 over the period [10, 20] at discount rate 5%:')
print(Econ.pv_continuous_payment(payment=10, discount_rate=0.05, discount_period=(10, 20)))
print('\nPresent value of a continuous payment of $10 over the period [10, 20] at discount rate 0%:')
print(Econ.pv_continuous_payment(payment=10, discount_rate=0, discount_period=(10, 20)))

print('\nEquivalent annual value of $50 over 10 years at discount rate 5%:')
print(Econ.equivalent_annual_value(present_value=50, discount_rate=0.05, discount_period=10))
