import sys, os
sys.path.append('../HPM573_SupportLib/scr')

import StatisticalClasses as sc
import numpy as numpy

# generate sample data
x = numpy.random.normal(10, 4, 1000)
y = numpy.random.normal(5, 8, 1000)

# check error raise
x2 = numpy.array([1,2,3])
y2 = numpy.array([-2,1,1])

# since the calling of mean and stdev functions here,
# if mean(y) == 0, the RatioStatIndp object can not be define
c = sc.RatioStatIndp('test',x2,y2)

# while it's ok for RatioStatPaired
c = sc.RatioStatPaired('test',x2,y2)

# except in bootstrap function
print(c.get_bootstrap_CI(5,100,'for_ratio'))
print(c.get_bootstrap_CI(5,100,'for_mean'))


# other test under normal situation
a = sc.RatioStatIndp('test',x,y)
print(a.get_summary(5,2))

a = sc.RatioStatPaired('test', x, y)
print(a.get_summary(5,2))

b = sc.DifferenceStatPaired('test', x, y)
print(b.get_summary(5,2))

b = sc.DifferenceStatIndp('test', x, y)
print(b.get_summary(5,2))