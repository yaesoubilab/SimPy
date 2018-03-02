
def format_number(number, deci):
    """
    :param number: number to be formatted
    :param deci: number of decimal places
    :returns: the text of number with the specified significant digit
    """
    if number is None:
        return ''
    else:
        return '{:.{prec}f}'.format(number, prec=deci)


def format_interval(interval, deci):
    """
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :returns a text of form '(low, up)' where the numbers have the specified significant digits"""

    if interval is None:
        return '(,)'
    else:
        return '({low:.{prec}f}, {up:.{prec}f})'\
            .format(low=interval[0], up=interval[1], prec=deci)


def format_estimate_interval(estimate, interval, deci):
    """
    :return: text in the form 'estimate (l, u)' with the specified decimal places
    """
    return format_number(estimate, deci) + ' ' + format_interval(interval, deci)


