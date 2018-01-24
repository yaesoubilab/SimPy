import numpy

from scr import StatisticalClasses as Stat
from scr import SupportFunctions as Support

# generate sample data
x = numpy.random.normal(10, 4, 1000)
y = numpy.random.normal(5, 8, 1000)


def print_results(stat):
    print('   Average =', Support.format_number(stat.get_mean(), digits=3))
    print('   St Dev =', Support.format_number(stat.get_stdev(), digits=3))
    print('   Min =', Support.format_number(stat.get_min(), digits=3))
    print('   Max =', Support.format_number(stat.get_max(), digits=3))
    print('   Median =', Support.format_number(stat.get_percentile(50), digits=3))
    print('   95% Mean Confidence Interval (t-based) =',
          Support.format_interval(stat.get_t_CI(5), 3))
    print('   95% Mean Confidence Interval (bootstrap) =',
          Support.format_interval(stat.get_bootstrap_CI(5, 1000), 3))
    print('   95% Percentile Interval =',
          Support.format_interval(stat.get_PI(5), 3))


def summary_stat_test(data):

    # define a summary statistics
    sum_stat = Stat.SummaryStat('Test summary statistics',data)

    print('Testing summary statistics:')
    print_results(sum_stat)

summary_stat_test(x)


def difference_stat_indp_test(x, y):

    # define
    stat = Stat.DifferenceStatIndp('Test DifferenceStatIndp', x, y)

    print('Testing DifferenceStatIndp:')
    print_results(stat)

difference_stat_indp_test(x,y)


def difference_stat_paired_test(x, y):

    # define
    stat = Stat.DifferenceStatPaired('Test DifferenceStatPaired', x, y)

    print('Testing DifferenceStatPaired:')
    print_results(stat)

difference_stat_paired_test(x,y)



def ratio_stat_indp_test(x, y):

    # define
    stat = Stat.RatioStatIndp('Test RatioStatIndp', x, y)

    print('Testing RatioStatIndp:')
    print_results(stat)

ratio_stat_indp_test(x,y)


def ratio_stat_paired_test(x, y):

    # define
    stat = Stat.RatioStatPaired('Test RatioStatPaired', x, y)

    print('Testing RatioStatPaired:')
    print_results(stat)

ratio_stat_paired_test(x,y)








# following functions are for discrete and continuous classes, unchanged
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