import math


def format_number(number, deci=None, sig_digits=None, format=None):
    """
    :param number: number to be formatted
    :param deci: number of decimal places
    :param sig_digits: number of significant digits
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :returns: the text of number with the specified format
    """

    if number is None:
        return ''

    # deci should be an integer
    deci, sig_digits = _get_deci_dig_digit(deci=deci, sig_digits=sig_digits)

    if format not in (None, ',', '%', '$', ''):
        raise ValueError('Invalid value for format.')

    if deci is not None:
        if format is None or format == '' or format is math.nan:
            return '{:.{prec}f}'.format(number, prec=deci)
        elif format == ',':
            return '{:,.{prec}f}'.format(number, prec=deci)
        elif format == '$':
            return '${:,.{prec}f}'.format(number, prec=deci)
        elif format == '%':
            return '{:.{prec}f}%'.format(100*number, prec=deci)

    elif sig_digits is not None:
        if format is None or format == '' or format is math.nan:
            return '{:.{prec}g}'.format(number, prec=sig_digits)
        elif format == ',':
            return '{:,.{prec}g}'.format(number, prec=sig_digits)
        elif format == '$':
            return '${:,.{prec}g}'.format(number, prec=sig_digits)
        elif format == '%':
            return '{:.{prec}g}%'.format(100*number, prec=sig_digits)
    else:
        return str(number)


def format_interval(interval, deci=None, sig_digits=None, format=None):
    """
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param sig_digits: number of significant digits
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :returns a text of form '(low, up)' where the numbers have the specified format """

    if interval is None:
        return ''

    deci, sig_digits = _get_deci_dig_digit(deci=deci, sig_digits=sig_digits)

    if deci is not None:
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

    elif sig_digits is not None:
        if format is None or format == '':
            return '({low:.{prec}g}, {up:.{prec}g})' \
                .format(low=interval[0], up=interval[1], prec=sig_digits)
        elif format == ',':
            return '({low:,.{prec}g}, {up:,.{prec}g})' \
                .format(low=interval[0], up=interval[1], prec=sig_digits)
        elif format == '$':
            return '(${low:,.{prec}g}, ${up:,.{prec}g})' \
                .format(low=interval[0], up=interval[1], prec=sig_digits)
        elif format == '%':
            return '({low:.{prec}g}%, {up:.{prec}g}%)' \
                .format(low=interval[0]*100, up=interval[1]*100, prec=sig_digits)
    else:
        return str(interval)


def format_estimate_interval(estimate, interval, deci, sig_digits, format=None):
    """
    :param estimate: the estimate
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param sig_digits: number of significant digits
    :param format: additional formatting instruction. 
        Use ',' to format as number, '%' to format as percentage, and '$' to format as currency
    :return: text in the form 'estimate (l, u)' with the specified decimal places
    """
    return format_number(number=estimate, deci=deci, sig_digits=sig_digits, format=format) \
           + ' ' + format_interval(interval=interval, deci=deci, sig_digits=sig_digits, format=format)


def _get_deci_dig_digit(deci, sig_digits):
    if deci is not None:
        try:
            deci = int(deci)
        except ValueError:
            raise ValueError('deci should be integer or float.')
        except TypeError:
            raise ValueError('deci should be integer or float.')
    elif sig_digits is not None:
        try:
            sig_digits = int(sig_digits)
        except ValueError:
            raise ValueError('sig_digits should be integer or float.')
        except TypeError:
            raise ValueError('sig_digits should be integer or float.')
    else:
        return None, None

    return deci, sig_digits
