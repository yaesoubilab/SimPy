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

df1.get_value(x_value=[1, 0])

