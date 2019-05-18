import SimPy.DataFrames as df

# one dimension dataframe for a continuous variable
df1 = df.OneDimDataFrame(y_values=[10, 9, 20, 40], x_min=0, x_max=3, x_delta=1)

print('print 10: ', df1.get_value(x_value=0))
print('print 10: ', df1.get_value(x_value=0.5))
print('print 9: ', df1.get_value(x_value=1))
print('print 40: ', df1.get_value(x_value=5))
print('')

# one dimension dataframe for a categorical variable
df2 = df.OneDimDataFrame(y_values=[10, 9, 20, 40], x_min=0, x_max=3, x_delta='int')

print('print 10: ', df2.get_value(x_value=0))
print('print 9: ', df2.get_value(x_value=1))