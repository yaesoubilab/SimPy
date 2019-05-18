import SimPy.DataFrames as df
import SimPy.RandomVariantGenerators as RVGs
import numpy as np

rows = [
    [0, 0, 0, 0.12],
    [0, 0, 1, 0.06],
    [0, 1, 0, 0.17],
    [0, 1, 1, 0.08],
    [5, 0, 0, 0.09],
    [5, 0, 1, 0.25],
    [5, 1, 0, 0.11],
    [5, 1, 1, 0.12],
]

rng = RVGs.RNG(seed=1)
probDf = df.ProbDistDataFrame(rows=rows,
                              list_x_min=[0, 0, 0],
                              list_x_max=[5, 1, 1],
                              list_x_delta=[5, 'int', 'int'])
# get a sample
print(probDf.get_sample_indices(rng=rng))

# testing to make sure it works
counts = [0]*8
for i in range(5000):
    idx = probDf.get_sample_indices(rng=rng)
    counts[idx[0]*4 + idx[1]*2 + idx[2]] += 1

print(np.array(counts)/sum(counts))
