from SimPy.DataFrames import Pyramid
from SimPy.Plots import PopulationPyramids as P

"""
a pyramid to store population size by age and sex:
    age,   sex,      population
    0,     0,        10,
    0,     1,        20,
    5,     0,        30,
    5,     1,        40,
    10,    0,        50,
    10,    1,        60
"""

pyramid = Pyramid(list_x_min=[0, 0],
                  list_x_max=[10, 1],
                  list_x_delta=[5, 'int'])

# adding a person of age 1.2 and sex = 0
pyramid.record_increment(x_values=[1.2, 0], increment=1)
# adding 10 persons of age 5 and sex = 1
pyramid.record_increment(x_values=[5, 1], increment=10)
# removing 2 persons of age 5.2 and sex = 1
pyramid.record_increment(x_values=[5.2, 1], increment=-2)
# adding 1 persons of age 15 and sex = 0
pyramid.record_increment(x_values=[15, 0], increment=1)

print('Size of group with age 0-5 and sex 0:', pyramid.get_current_value(x_values=[0, 0]))
print('Size of group with age 5-10 and sex 1:', pyramid.get_current_value(x_values=[5, 1]))
# get the total population size
print('Population size:', pyramid.get_sum())
# get the size of each group
print('Population size by age, sex:', pyramid.get_values())
# get the percentage of population in each group
print('Population distribution by age, sex', pyramid.get_percentages())
# plot the pyramid
P.plot_pyramids(observed_data=pyramid.get_percentage(),
                simulated_data=None,
                x_lim=100,
                y_lim=12.5,
                title='Testing Pyramid')


# updating a new pyramid based on the one above
newPyramid = Pyramid(list_x_min=[0, 0],
                     list_x_max=[10, 1],
                     list_x_delta=[5, 'int'])
newPyramid.record_values_from_another_pyramid(another_pyramid=pyramid)

print('\nTesting the new pyramid:')
# get the total population size
print('Population size:', newPyramid.get_sum())
# get the size of each group
print('Population size by age, sex:', newPyramid.get_values())
# get the percentage of population in each group
print('Population distribution by age, sex', newPyramid.get_percentages())
