import SimPy.DataFrames as df
import SimPy.RandomVariantGenerators as RVGs
import numpy as np

rows = [
    [0,     0,  0.1],
    [0,     1,  0.125],
    [5,     0,  0.1],
    [5,     1,  0.25],
    [10,    0,  0.2],
    [10,    1,  0.225]
]

rng = RVGs.RNG(seed=1)
probDf = df.ProbDistDataFrame(rows=rows,
                              list_x_min=[0, 0],
                              list_x_max=[10, 1],
                              list_x_delta=[5, 'int'])
# get a sample
print(probDf.get_sample_indices(rng=rng))

# testing to make sure it works
counts = [0]*6
for i in range(5000):
    idx = probDf.get_sample_indices(rng=rng)
    counts[idx[0]*2 + idx[1]] += 1

print(np.array(counts)/sum(counts))
