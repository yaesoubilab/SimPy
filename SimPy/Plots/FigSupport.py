

def output_figure(plt, filename=None):
    """
    :param plt: reference to the plot
    :param filename: filename to save this figure as (e.g. 'figure.png') (if None, the figure will be displayed)
    """
    # output
    if filename is None:
        plt.show()
    else:
        try:
            plt.savefig(filename, dpi=300)
        except:
            raise ValueError('Error in saving figure ' + filename)
