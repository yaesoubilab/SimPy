

def proper_file_name(text):
    """
    :param text: filename
    :return: filename where invalid characters are removed
    """

    return text.replace('|', ',').replace(':', ',').replace('<', 'l').replace('>', 'g').replace('\n', '')


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
        try:
            plt.savefig(proper_file_name(filename), dpi=dpi)
        except:
            raise ValueError("Error in saving figure '{}'. Check the filename and the path.".format(filename))
