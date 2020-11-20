from SimPy.Support.MiscFunctions import *
import SimPy.FormatFunctions as F
import os


def output_figure(plt, filename=None, dpi=300):
    """
    :param plt: reference to the plot
    :param filename: filename to save this figure as (e.g. 'figure.png') (if None, the figure will be displayed)
    :param dpi: dpi of the figure
    """
    # output
    if filename is None:
        plt.show()
    else:
        if filename[0] == '/':
            filename = filename[1:]
        # get directory
        directory_path = os.path.dirname(filename)

        # create the directory if does not exist
        if directory_path != '':
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

        try:
            plt.savefig(proper_file_name(filename), dpi=dpi)
        except ValueError:
            raise ValueError("Error in saving figure '{}'. "
                             "Ensure that the filename is valid and "
                             "that the folder where the figure should be saved exists.".format(filename))


def calculate_ticks(l, u, delta):
    values = []
    v = l
    while v <= u:
        values.append(v)
        v += delta

    return values


def format_x_axis(ax, min_x, max_x, delta_x, buffer=0, form=None, deci=None):

    # get x ticks
    xs = ax.get_xticks()

    if min_x is None:
        min_x = xs[0]
    if max_x is None:
        max_x = xs[-1]
    if delta_x is not None:
        xs = calculate_ticks(min_x, max_x, delta_x)

    # format x-axis
    ax.set_xticks(xs)
    if deci is not None:
        ax.set_xticklabels([F.format_number(x, deci=deci, format=form) for x in xs])

    ax.set_xlim([min_x-buffer, max_x+buffer])
