import StatisticalClasses as sc
import numpy as numpy

# generate independent sample data
x_i = numpy.random.normal(10, 4, 1000)
y_i = numpy.random.normal(5, 8, 1000)

# generate paired sample data
x_p = numpy.random.normal(10, 4, 1000)
y_p = x_p + numpy.random.normal(-5, 1, 1000)

# get warnings if there is 0 in the denominator variable
x = numpy.array([1,2,3])
y = numpy.array([0,1,2])


