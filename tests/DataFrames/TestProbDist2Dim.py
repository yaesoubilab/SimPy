import SimPy.DataFrames as df
import SimPy.RandomVariantGenerators as RVGs
import numpy as np
import math

rows = [
    [0,     0,  0.1],
    [0,     1,  0.125],
    [5,     0,  0.1],
    [5,     1,  0.25],
    [10,    0,  0.2],
    [10,    1,  0.225]
]

rng = RVGs.RNG(seed=1)
probDf = df.DataFrameWithEmpiricalDist(rows=rows,
                                       list_x_min=[0, 0],
                                       list_x_max=[10, 1],
                                       list_x_delta=[5, 'int'])
# get a sample
print('Get a sampled index:', probDf.sample_indices(rng=rng))
print('Get a sampled index:', probDf.sample_values(rng=rng))
print('')

# testing to make sure sample by index works
counts = [0]*6
for i in range(5000):
    idx = probDf.sample_indices(rng=rng)
    counts[idx[0]*2 + idx[1]] += 1

print('Testing the sampling by index:')
print(np.array(counts)/sum(counts))
print('')

# testing to make sure sample by value is working
for i in range(5000):
    values = probDf.sample_values(rng=rng)
    i_0 = math.floor(values[0]/5)
    counts[i_0*2 + values[1]] += 1

print('Testing the sampling by value:')
print(np.array(counts)/sum(counts))