
def format_number(number, deci, format=None):
    """
    :param number: number to be formatted
    :param deci: number of decimal places
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :returns: the text of number with the specified format
    """
    if number is None:
        return ''
    else:
        if format is None or format == '':
            return '{:.{prec}f}'.format(number, prec=deci)
        elif format == ',':
            return '{:,.{prec}f}'.format(number, prec=deci)
        elif format == '$':
            return '${:,.{prec}f}'.format(number, prec=deci)
        elif format == '%':
            return '{:.{prec}f}%'.format(100*number, prec=deci)


def format_interval(interval, deci, format=None):
    """
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :returns a text of form '(low, up)' where the numbers have the specified format """

    if interval is None:
        return ''
    else:
        if format is None or format == '':
            return '({low:.{prec}f}, {up:.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif format == ',':
            return '({low:,.{prec}f}, {up:,.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif format == '$':
            return '(${low:,.{prec}f}, ${up:,.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif format == '%':
            return '({low:.{prec}f}%, {up:.{prec}f}%)' \
                .format(low=interval[0]*100, up=interval[1]*100, prec=deci)


def format_estimate_interval(estimate, interval, deci, format=None):
    """
    :param estimate: the estimate
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :return: text in the form 'estimate (l, u)' with the specified decimal places
    """
    return format_number(estimate, deci, format) + ' ' + format_interval(interval, deci, format)


