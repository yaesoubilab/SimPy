import sys, os
sys.path.append('../HPM573_SupportLib/scr')

import StatisticalClasses as sc
import numpy as numpy

# generate sample data
x = numpy.random.normal(10, 4, 1000)
y = numpy.random.normal(5, 8, 1000)

# test
a = sc.RatioStatIndp('test',x,y)
print(a.get_summary(5,2))

a = sc.RatioStatPaired('test', x, y)
print(a.get_summary(5,2))

b = sc.DifferenceStatPaired('test', x, y)
print(b.get_summary(5,2))

b = sc.DifferenceStatIndp('test', x, y)
print(b.get_summary(5,2))