import SimPy.DataFrames as df
import SimPy.RandomVariantGenerators as RVGs

# last column includes the rate (e.g. mortality rate) of
# exponential distributions that should be created for each group.
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

df2 = df.DataFrameWithExpDist(rows=rows,
                              list_x_min=[0, 0, 0],
                              list_x_max=[5, 1, 1],
                              list_x_delta=[5, 'int', 'int'])

rng = RVGs.RNG(seed=1)
print('Sample 1: ', df2.get_dist(x_value=[0, 0, 0]).sample(rng))
print('Sample 2', df2.get_dist(x_value=[1, 0, 0]).sample(rng))
print('Sample 3', df2.get_dist(x_value=[2, 1, 0]).sample(rng))
print('Sample 4', df2.get_dist(x_value=[5, 0, 1]).sample(rng))
