import SimPy.EconEval as Econ

a = Econ._assert_np_list([1, 2], 'error')
print(a)

b = Econ._assert_np_list('test', 'error')
print(b)