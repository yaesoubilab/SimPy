import SimPy.DataFrames as df
import SimPy.RandomVariantGenerators as RVGs
import numpy as np
import math

# rows = [
#     [0,     0,  0.1],
#     [0,     1,  0.125],
#     [5,     0,  0.1],
#     [5,     1,  0.25],
#     [10,    0,  0.2],
#     [10,    1,  0.225]
# ]

rows = [
    [0, 0, 0.030],  # < 5, male
    [0, 1, 0.030],  # < 5, female
    [5, 0, 0.031],  # 5-9, male
    [5, 1, 0.031],  # 5-9, female
    [10, 0, 0.033],  # 10-14, male
    [10, 1, 0.031],  # 10-14, female
    [15, 0, 0.033],  # 15-19, male
    [15, 1, 0.031],  # 15-19, female
    [20, 0, 0.035],  # 20-24, male
    [20, 1, 0.033],  # 20-24, female
    [25, 0, 0.037],  # 25-29, male
    [25, 1, 0.035],  # 25-29, female
    [30, 0, 0.034],  # 30-34, male
    [30, 1, 0.033],  # 30-34, female
    [35, 0, 0.033],  # 35-39, male
    [35, 1, 0.031],  # 35-39, female
    [40, 0, 0.030],  # 40-44, male
    [40, 1, 0.030],  # 40-44, female
    [45, 0, 0.032],  # 45-49, male
    [45, 1, 0.033],  # 45-49, female
    [50, 0, 0.032],  # 50-54, male
    [50, 1, 0.033],  # 50-54, female
    [55, 0, 0.032],  # 55-59, male
    [55, 1, 0.035],  # 55-59, female
    [60, 0, 0.029],  # 60-64, male
    [60, 1, 0.031],  # 60-64, female
    [65, 0, 0.024],  # 65-69, male
    [65, 1, 0.027],  # 65-69, female
    [70, 0, 0.018],  # 70-74, male
    [70, 1, 0.027],  # 70-74, female
    [75, 0, 0.012],  # 75-79, male
    [75, 1, 0.015],  # 75-79, female
    [80, 0, 0.008],  # 80-84, male
    [80, 1, 0.011],  # 80-84, female
    [85, 0, 0.007],  # >= 85, male
    [85, 1, 0.013]  # >= 85, female
]

rng = RVGs.RNG(seed=1)
probDf = df.DataFrameWithEmpiricalDist(rows=rows,
                                       list_x_min=[0, 0],
                                       list_x_max=[85, 1],
                                       list_x_delta=[5, 'int'])
# get a sample
print('Get a sampled index:', probDf.sample_indices(rng=rng))
print('Get a sampled index:', probDf.sample_values(rng=rng))
print('')

# testing to make sure sample by index works
counts = [0]*len(rows)
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
