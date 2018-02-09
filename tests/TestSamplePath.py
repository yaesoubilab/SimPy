from scr import SamplePathClass as Path

# a sample path with initial size = 1
mySamplePath = Path.SamplePath('my sample path', 0, 1)

# record the observations
mySamplePath.record(1.5, 2)
mySamplePath.record(2, -1)
mySamplePath.record(5, 0)

# plot
Path.graph_sample_path(mySamplePath, 'Test', 'time', 'y', Path.OutType.SHOW, "Test")


# second sample path with initial size = 1
mySamplePath2 = Path.SamplePath('my sample path 2', 0, 1)

# record the observations
mySamplePath2.record(0.5, 4)
mySamplePath2.record(1.8, -2)
mySamplePath2.record(5.5, 1)

Path.graph_sample_paths(
    sample_paths=[mySamplePath, mySamplePath2],
    title='Test 2 sample paths',
    x_label='time',
    y_label='y',
    output_type=Path.OutType.SHOW,
    legends=['S1', 'S2'],
    transparency=0.5,
    common_color_code='g')
