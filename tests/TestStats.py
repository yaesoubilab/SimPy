import random
from scr import StatisticalClasses as Stat
from scr import SupportFunctions as Support


def print_results(stat):
    print('   Average =', Support.format_number(stat.get_mean(), digits=3))
    print('   St Dev =', Support.format_number(stat.get_stdev(), digits=3))
    print('   Min =', Support.format_number(stat.get_min(), digits=3))
    print('   Max =', Support.format_number(stat.get_max(), digits=3))
    print('   Median =', Support.format_number(stat.get_percentile(50), digits=3))
    print('   95% Mean Confidence Interval (t-based) =',
          Support.format_interval(stat.get_t_CI(.05), 3))
    print('   95% Mean Confidence Interval (bootstrap) =',
          Support.format_interval(stat.get_bootstrap_CI(.05, 1000), 3))
    print('   95% Prediction Interval =',
          Support.format_interval(stat.get_PI(.05), 3))


def summary_stat_test(data):

    # define a summary statistics
    sum_stat = Stat.SummaryStat('Test summary statistics',data)

    print('Testing summary statistics:')
    print_results(sum_stat)


def discrete_time_test(data):

    # define a summary statistics
    discrete_stat = Stat.DiscreteTimeStat('Test discrete-time statistics')

    # record data points
    for point in data:
        discrete_stat.record(point)

    print('Testing discrete-time statistics:')
    print_results(discrete_stat)


def continuous_time_test(times, observations):

    continuous_stat = Stat.ContinuousTimeStat('Test continuous-time statistics', 0)

    for obs in range(0, len(times)):
        # find the increment
        inc = 0
        if obs == 0:
            inc = observations[obs]
        else:
            inc = observations[obs] - observations[obs - 1]
        continuous_stat.record(times[obs], inc)

    print('Testing continuous-time statistics:')
    print_results(continuous_stat)


# populate a data set to test offline and online summary statistics
samples = []
for i in range(0, 1000):
    samples.append(random.uniform(-100, 100))

summary_stat_test(samples)
discrete_time_test(samples)

# populate a data set to test continuous-time statistics
sampleT = []
sampleObs = []
i = 0
for i in range(0, 100):
    t = random.uniform(i, i + 1)
    sampleT.append(t)
    sampleObs.append(10*t)

continuous_time_test(sampleT, sampleObs)