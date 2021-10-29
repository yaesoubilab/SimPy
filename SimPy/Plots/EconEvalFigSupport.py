
def format_ax(ax,
              x_range=None, x_delta=None,
              y_range=None, delta_y=None, if_y_axis_prob=True,
              if_format_y_numbers=True, y_axis_decimal=1):

    # the range of x and y-axis are set so that we can get the
    # tick values and label
    if y_range is None and if_y_axis_prob:
        ax.set_ylim((-0.01, 1.01))
    if y_range:
        ax.set_ylim(y_range)
    if x_range:
        ax.set_xlim(x_range)

    # get x ticks
    if x_delta is None:
        vals_x = ax.get_xticks()
    else:
        vals_x = []
        x = x_range[0]
        while x <= x_range[1]:
            vals_x.append(x)
            x += x_delta

    # get y ticks
    if delta_y is None:
        vals_y = ax.get_yticks()
    else:
        vals_y = []
        y = y_range[0]
        while y <= y_range[1]:
            vals_y.append(y)
            y += delta_y

    # format x-axis
    ax.set_xticks(vals_x)
    ax.set_xticklabels(['{:,.{prec}f}'.format(x, prec=0) for x in vals_x])

    d = 2 * (x_range[1] - x_range[0]) / 200
    ax.set_xlim([x_range[0] - d, x_range[1] + d])

    # format y-axis
    if y_range is None:
        ax.set_yticks(vals_y)
    if if_y_axis_prob:
        ax.set_yticklabels(['{:.{prec}f}'.format(x, prec=1) for x in vals_y])
    elif if_format_y_numbers:
        ax.set_yticklabels(['{:,.{prec}f}'.format(x, prec=y_axis_decimal) for x in vals_y])

    if y_range is None and if_y_axis_prob:
        ax.set_ylim((-0.01, 1.01))
    if y_range:
        ax.set_ylim(y_range)

    if not if_y_axis_prob:
        ax.axhline(y=0, c='k', ls='--', linewidth=0.5)


def add_curves_to_ax(ax, curves, legends, x_range, x_delta, y_range, show_legend,
                     line_width, opt_line_width, legend_font_size):

    for i, curve in enumerate(curves):
        # plot line
        if legends is None:
            ax.plot(curve.xs, curve.ys, c=curve.color, alpha=1,
                    label=curve.label, linewidth=line_width)
        else:
            ax.plot(curve.xs, curve.ys, c=curve.color, alpha=1,
                    label=legends[i], linewidth=line_width)
        ax.plot(curve.optXs, curve.optYs, c=curve.color, linewidth=opt_line_width)

    format_ax(ax=ax, y_range=y_range,
              x_range=x_range,
              x_delta=x_delta,
              if_y_axis_prob=True)

    if show_legend:
        ax.legend(fontsize=legend_font_size)  # xx-small, x-small, small, medium, large, x-large, xx-large
