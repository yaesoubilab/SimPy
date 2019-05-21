import SimPy.DataFrames as df

rows = [
    [0,     0,  10],
    [0,     1,  20],
    [5,     0,  11],
    [5,     1,  12],
    [10,    0,  15],
    [10,    1,  16]
]

df1 = df.DataFrame(rows=rows,
                   list_x_min=[0, 0],
                   list_x_max=[10, 1],
                   list_x_delta=[5, 'int'])

print('print rows:')
print(df1.get_rows())
print(df1.get_objs())

for obj in df1.get_objs_gen():
    print(obj)
print('')

print('print: 10', df1.get_value(x_value=[0, 0]))
print('print: 10', df1.get_value(x_value=[1, 0]))
print('print: 20', df1.get_value(x_value=[2, 1]))
print('print: 12', df1.get_value(x_value=[5, 1]))
print('print: 15', df1.get_value(x_value=[9, 0]))
print('print: 16', df1.get_value(x_value=[10, 1]))
print('print: 16', df1.get_value(x_value=[20, 1]))
print('')

rows = [
    [0,     0,  0,  10],
    [0,     0,  1,  11],
    [0,     1,  0,  20],
    [0,     1,  1,  21],
    [5,     0,  0,  30],
    [5,     0,  1,  31],
    [5,     1,  0,  40],
    [5,     1,  1,  41]
]

df2 = df.DataFrame(rows=rows,
                   list_x_min=[0, 0, 0],
                   list_x_max=[5, 1, 1],
                   list_x_delta=[5, 'int', 'int'])

print('print rows:')
print(df2.get_rows())
print('')

print('print: 10', df2.get_value(x_value=[0, 0, 0]))
print('print: 10', df2.get_value(x_value=[1, 0, 0]))
print('print: 20', df2.get_value(x_value=[2, 1, 0]))
print('print: 31', df2.get_value(x_value=[5, 0, 1]))
print('print: 30', df2.get_value(x_value=[9, 0, 0]))
print('print: 41', df2.get_value(x_value=[9, 1, 1]))
