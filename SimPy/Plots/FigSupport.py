

def output_figure(plt, output_type='show', filename='figure'):
    """
    :param plt: reference to the plot
    :param output_type: select from 'show', 'png', or 'pdf'
    :param filename: filename to save this figure as
    """
    # output
    if output_type == 'show':
        plt.show()
    elif output_type == 'png':
        plt.savefig(filename + ".png")
    elif output_type == 'pdf':
        plt.savefig(filename + ".pdf")
    else:
        raise ValueError("Invalid value for figure output type. "
                         "Valid values are: 'show', 'png', and 'pdf' ")


def get_moving_average(data, window=2):
    """
    calculates the moving average of a time-series
    :param data: list of observations
    :param window: the window (number of data points) over which the average should be calculated
    :return: list of moving averages
    """

    if window < 2:
        raise ValueError('The window over which the moving averages '
                         'should be calculated should be greater than 1.')
    if window >= len(data):
        raise ValueError('The window over which the moving averages '
                         'should be calculated should be less than the number of data points.')

    averages = []

    # the 'window - 1' averages cannot be calculated
    for i in range(window-1):
        averages.append(None)

    # calculate the first moving average
    moving_ave = sum(data[0:window])/window
    averages.append(moving_ave)

    for i in range(window, len(data)):
        moving_ave = (moving_ave*window - data[i-window] + data[i])/window
        averages.append(moving_ave)

    return averages

