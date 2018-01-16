from scr import SamplePathClass as Path

# a sample path with initial size = 1
mySamplePath = Path.SamplePath('my sample path', 0, 1)

# record the observations
mySamplePath.record(1.5, 2)
mySamplePath.record(2, -1)
mySamplePath.record(5, 0)

# plot
Path.graph_sample_path(mySamplePath, 'Test', 'time', 'y', Path.OutType.SHOW)

