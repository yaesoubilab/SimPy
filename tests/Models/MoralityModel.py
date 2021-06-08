import SimPy.RandomVariateGenerators as RVGs
from SimPy.Models import MortalityModel

# from life tables
rows = [
    # age, sex , mortality rate
    [0,     0,  0.1],
    [0,     1,  0.125],
    [5,     0,  0.1],
    [5,     1,  0.25],
    [10,    0,  0.2],
    [10,    1,  0.225]
]

rng = RVGs.RNG(seed=1)

# mortality model
mortalityModel = MortalityModel(mortality_rates=rows,  # life table
                                group_mins=0,  # minimum value of sex group
                                group_maxs=1,  # maximum value of sex group
                                group_delta='int',  # sex group is a category
                                age_min=0,  # minimum age in this life table
                                age_delta=5)         # age interval

# get sample for time until death
print(mortalityModel.sample_time_to_death(group=0, age=8.9, rng=rng))
print(mortalityModel.sample_time_to_death(group=0, age=0, rng=rng))
print(mortalityModel.sample_time_to_death(group=0, age=32, rng=rng))


