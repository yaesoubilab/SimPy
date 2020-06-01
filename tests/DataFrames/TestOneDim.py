import SimPy.DataFrames as df
import SimPy.RandomVariateGenerators as RVGs

# one dimension dataframe for a continuous variable
df1 = df.OneDimDataFrame(y_objects=[10, 9, 20, 40], x_min=0, x_max=3, x_delta=1)

print('Testing one dimensional data frame with a continuous variable:')
print('print 10: ', df1.get_obj(x_value=0))
print('print 10: ', df1.get_obj(x_value=0.5))
print('print 9: ', df1.get_obj(x_value=1))
print('print 40: ', df1.get_obj(x_value=5))
print('print 40: ', df1.get_obj(x_value=3))
print('')

# one dimension dataframe for a categorical variable
df2 = df.OneDimDataFrame(y_objects=[10, 9, 20, 40], x_min=0, x_max=3, x_delta='int')

print('Testing one dimensional data frame with a categorical variable:')
print('print 10: ', df2.get_obj(x_value=0))
print('print 9: ', df2.get_obj(x_value=1))
print('')

# one dimension dataframe for a continuous variable
# assuming that y_values are rates of exponential distributions
df1 = df.OneDimDataFrameWithExpDist(y_values=[10, 9, 20, 40], x_min=0, x_max=3, x_delta=1)

rng = RVGs.RNG(seed=1)
print('Testing one dimensional data frame with exponential distributions')
print('Sample 1: ', df1.get_dist(x_value=0).sample(rng))
print('Sample 2: ', df1.get_dist(x_value=0.5).sample(rng))
print('Sample 3: ', df1.get_dist(x_value=1).sample(rng))
print('Sample 4: ', df1.get_dist(x_value=5).sample(rng))
print('')
