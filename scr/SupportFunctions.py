
def format_number(number, digits):
    """
    :param number: number to be formatted
    :param digits: number of significant digits
    :returns: the text of number with the specified significant digit
    """
    if number is None:
        return ''
    else:
        return '{:.{prec}f}'.format(number, prec=digits)


def format_interval(interval, digits):
    """
    :param interval: list of form [low up]
    :param digits: number of significant digits
    :returns a text of form '(low, up)' where the numbers have the specified significant digits"""

    if interval is None:
        return '(,)'
    else:
        return '({low:.{prec}f}, {up:.{prec}f})'\
            .format(low=interval[0], up=interval[1], prec=digits)


def format_estimate_interval(estimate, interval, digits):
    """
    :return: text in the form 'estimate (l, u)'
    """
    return format_number(estimate, digits) + ' ' + format_interval(interval, digits)


