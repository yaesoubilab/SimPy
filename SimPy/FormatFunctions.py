from enum import Enum


class FormatNumber(Enum):
    NUMBER = 0          # 1,234
    CURRENCY = 1        # $1,234
    PERCENTAGE = 2      # 1.23%


def format_number(number, deci, form=None):
    """
    :param number: number to be formatted
    :param deci: number of decimal places
    :param form: additional formatting instruction.
        Can take values from Format.NUMBER, Format.CURRENCY, and Format.PERCENTAGE
    :returns: the text of number with the specified format
    """
    if number is None:
        return ''
    else:
        if form is None:
            return '{:.{prec}f}'.format(number, prec=deci)
        elif form == FormatNumber.NUMBER:
            return '{:,.{prec}f}'.format(number, prec=deci)
        elif form == FormatNumber.CURRENCY:
            return '${:,.{prec}f}'.format(number, prec=deci)
        elif form == FormatNumber.PERCENTAGE:
            return '{:.{prec}f}%'.format(100*number, prec=deci)


def format_interval(interval, deci, form=None):
    """
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param form: additional formatting instruction.
        Can take values from Format.NUMBER, Format.CURRENCY, and Format.PERCENTAGE
    :returns a text of form '(low, up)' where the numbers have the specified format """

    if interval is None:
        return '(,)'
    else:
        if form is None:
            return '({low:.{prec}f}, {up:.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif form == FormatNumber.NUMBER:
            return '({low:,.{prec}f}, {up:,.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif form == FormatNumber.CURRENCY:
            return '(${low:,.{prec}f}, ${up:,.{prec}f})' \
                .format(low=interval[0], up=interval[1], prec=deci)
        elif form == FormatNumber.PERCENTAGE:
            return '({low:.{prec}f}%, {up:.{prec}f}%)' \
                .format(low=interval[0]*100, up=interval[1]*100, prec=deci)


def format_estimate_interval(estimate, interval, deci, form=None):
    """
    :param estimate: the estimate
    :param interval: list of form [low up]
    :param deci: number of decimal places
    :param form: additional formatting instruction.
        Can take values from Format.NUMBER, Format.CURRENCY, and Format.PERCENTAGE
    :return: text in the form 'estimate (l, u)' with the specified decimal places
    """
    return format_number(estimate, deci, form) + ' ' + format_interval(interval, deci, form)


