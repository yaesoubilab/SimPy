from SimPy import SamplePathClasses as Path

# a sample path with initial size = 1
path1 = Path.PrevalencePathRealTimeUpdate('Path 1', 0, sim_rep=1)
# record the observations
path1.record(1.5, 2)
path1.record(2, -1)
path1.record(5, 0)

# second sample path with initial size = 1
path2 = Path.PrevalencePathRealTimeUpdate('Path 2', 0, sim_rep=1)
# record the observations
path2.record(0.5, 4)
path2.record(1.8, -2)
path2.record(5.5, 1)

# third sample path with initial size = 1
path3 = Path.PrevalencePathBatchUpdate(
    'Path 3', 0, times_of_changes=[1.5, 2, 5], increments=[2, -1, 0], sim_rep=1)


# plot path 1 only
Path.graph_sample_path(
    sample_path=path1,
    title='Plotting a single sample path',
    x_label='time',
    y_label='observed value',
    legend= 'Path 1',
    color_code='r')

# plot path 3 only
Path.graph_sample_path(
    sample_path=path3,
    title='Plotting a single sample path that is updated in batch',
    x_label='time',
    y_label='observed value',
    legend= 'Path 3',
    color_code='r')

# plot both paths
Path.graph_sample_paths(
    sample_paths=[path1, path2],
    title='Plot two sample paths with different color',
    x_label='time',
    y_label='observed value',
    legends=['Path 1', 'Path 2'],
    transparency=0.75)

Path.graph_sample_paths(
    sample_paths=[path1, path2],
    title='Plot 2 sample paths with the same color',
    x_label='time',
    y_label='observed value',
    legends='Path',
    transparency=0.5,
    common_color_code='g')
