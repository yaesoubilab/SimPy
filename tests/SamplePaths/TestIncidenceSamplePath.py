from SimPy import SamplePathClasses as Path

# a sample path with initial size = 1
path1 = Path.IncidenceSamplePath('Path 1', delta_t=1, sim_rep=1)
# record the observations
path1.record_increment(1.2, 2)
path1.record_increment(2.3, 1)
path1.record_increment(5.1, 5)
path1.close(6)
# stats
print(path1.stat.get_mean())

# second sample path with initial size = 1
path2 = Path.IncidenceSamplePath('Path 2', delta_t=1, sim_rep=1)
# record the observations
path2.record_increment(0.5, 4)
path2.record_increment(1.8, 2)
path2.record_increment(5.5, 1)
path2.close(6)
# stats
print(path2.stat.get_mean())

# plot path 1 only
Path.graph_sample_path(
    sample_path=path1,
    title='Plotting a single sample path',
    x_label='Time Period',
    y_label='observed value',
    legend='Path 1',
    color_code='r',
    connect='line')

Path.graph_sample_path(
    sample_path=path2,
    title='Plotting a single sample path',
    x_label='Time Period',
    y_label='observed value',
    legend='Path 2',
    color_code='r',
    connect='line')
