from SimPy.DataFrames import Pyramid

pyramid = Pyramid(list_x_min=[0, 0], list_x_max=[10, 1], list_x_delta=[5, 'int'])

pyramid.record_increment(x_values=[1.2, 0], increment=1)
pyramid.record_increment(x_values=[2, 1], increment=1)
pyramid.record_increment(x_values=[15, 0], increment=1)


print(pyramid.get_current_value(x_values=[1, 0]))

print(pyramid.get_table_of_values())
print(pyramid.get_sum())
print(pyramid.get_percentage())
