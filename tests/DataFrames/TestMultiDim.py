import SimPy.DataFrames as df

rows = [
    [0,     0,  10],
    [0,     1,  20],
    [5,     0,  11],
    [5,     1,  12],
    [10,    0,  15],
    [10,    1,  16]
]

df1 = df.MultiDimDataFrame(rows=rows,
                           list_x_min=[0, 0],
                           list_x_max=[10, 1],
                           list_x_delta=[5, 1])

print('print: 10', df1.get_value(x_value=[0, 0]))
print('print: 10', df1.get_value(x_value=[1, 0]))
print('print: 20', df1.get_value(x_value=[2, 1]))
print('print: 12', df1.get_value(x_value=[5, 1]))
print('print: 15', df1.get_value(x_value=[9, 0]))
print('print: 16', df1.get_value(x_value=[20, 1]))


