import matplotlib.pyplot as plt
import numpy as np


def plot_pyramids(observed_data, simulated_data=None,
                  title=None, fig_size=None,
                  x_lim=100,
                  y_labels=None,
                  age_group_width=5,
                  colors=('brown', 'grey', 'blue'),
                  legend=('Data', 'Model'),
                  length_of_sim_bars=500,
                  linewidths_of_sim_bars=1,
                  scale_of_sim_legend=0.5,
                  transparency=0.5,
                  file_name=None):
    """
    :param observed_data: example:
                        [[0, 0, 0.1], [0, 1, 0.2], [5, 0, 0.3], [5, 1, 0.4], [10, 0, 0.6], [10, 1, 0.4]]
    :param simulated_data: example:
                        [
                            [[0, 0, 0.5], [0, 1, 0.3], [5, 0, 0.5], [5, 1, 0.6]],
                            [[0, 0, 0.2], [0, 1, 0.4], [5, 0, 0.9], [5, 1, 0.4]]
                        ]
    :param title: (string) figure title
    :param fig_size: (tuple) figure size (e.g. (2, 4))
    :param x_lim: maximum of x-axis (between 0 and 100)
    :param y_labels: (list) of strings for labels of y_axis
    :param age_group_width: width of each age group in years.
    :param colors: (tuple) example: ('brown', 'grey', 'blue')
                                    for men, women and simulation bars
    :param legend: (tuple) default ('Data', 'Model').
                    legend will not be displayed if set to None
    :param length_of_sim_bars: length of simulation bars
    :param linewidths_of_sim_bars: line width of the simulation bars
    :param scale_of_sim_legend: (between 0 and 1) to shrink simulation bars
                                shown on the legends
    :param transparency: transparency of bars
    :param file_name: (string) to save the figure. Example: 'Pyramid.png'.
                    if not provided, the figure will be displayed.
    """

    w_sizes = []    # sizes of female age groups
    m_sizes = []    # sizes of male age groups
    w_ages = []     # age breaks for women
    m_ages = []     # age breaks for men

    # number of sex and age groups
    num_of_groups = len(observed_data)

    i = 0
    # sort data into separate arrays
    while i < num_of_groups:
        # woman
        if observed_data[i][1] == 1:
            w_sizes.append(observed_data[i][2] * 100)
            w_ages.append(observed_data[i][0])
        # men
        else:
            m_sizes.append(observed_data[i][2] * 100)
            m_ages.append(observed_data[i][0])
        i = i + 1

    if m_ages != w_ages:
        raise ValueError('Male and female age groups should be the same.')

    # find maximum value of the y_axis
    y_lim = m_ages[-1] - m_ages[0] + age_group_width / 2

    # either sets to default ticks or user input
    y_ticks_places = range(0, len(m_ages) * age_group_width, age_group_width)

    # either sets to default tick labels or user input
    if y_labels is None:
        y_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
                    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
                    '80-84', '85-89', '90-94', '95-100', '100+']

    # plot configuration
    fig, axis = plt.subplots(ncols=2, sharey=True, tight_layout=True, figsize=fig_size)
    # figure title
    st = fig.suptitle(title)
    # add y labels and ticks
    plt.setp(axis, yticks=y_ticks_places, yticklabels=y_labels[:len(y_ticks_places)])

    # ranges of x- and y-axes
    for i in [0, 1]:
        axis[i].xmin, axis[i].xmax, axis[i].ymin, axis[i].ymax \
            = axis[i].axis([0, x_lim, -age_group_width * (1 / 2 + 1 / 10), y_lim + age_group_width * 1 / 10])

    # bar plot for males
    axis[0].barh(np.array(m_ages)-m_ages[0], m_sizes, age_group_width,
                 align='center', color=colors[0],
                 edgecolor='black', alpha=transparency, label='Data')
    axis[0].invert_xaxis()

    # bar plot for females
    axis[1].barh(np.array(w_ages)-w_ages[0], w_sizes, age_group_width,
                 align='center', color=colors[1],
                 edgecolor='black', alpha=transparency, label='Data')

    # title of each sub-figure
    axis[0].set(title='\nMen')
    axis[1].set(title='\nWomen')

    # add labels
    axis[0].set_ylabel('Age')
    axis[0].set_xlabel('Percent of Population')
    axis[1].set_xlabel('Percent of Population')

    # adding pyramids from simulation
    if simulated_data is not None:
        n_replications = len(simulated_data)
        rep = 0     # simulation replication
        g = 0       # age-sex group
        sim_data_w = []
        sim_data_m = []
        sim_data_wa = []
        sim_data_ma = []
        while rep < n_replications:
            while g < len(simulated_data[rep]):
                if simulated_data[rep][g][1] == 1:
                    sim_data_w.append(simulated_data[rep][g][2] * 100)
                    sim_data_wa.append(simulated_data[rep][g][0] - w_ages[0])
                    g = g + 1
                else:
                    sim_data_m.append(simulated_data[rep][g][2] * 100)
                    sim_data_ma.append(simulated_data[rep][g][0] - m_ages[0])
                    g = g + 1
            g = 0
            rep = rep + 1
        axis[0].scatter(sim_data_m, sim_data_ma,
                        marker='|', linewidths=linewidths_of_sim_bars,
                        s=length_of_sim_bars,
                        color=colors[2], alpha=transparency,
                        label='Model')
        axis[1].scatter(sim_data_w, sim_data_wa,
                        marker='|', linewidths=linewidths_of_sim_bars,
                        s=length_of_sim_bars,
                        color=colors[2], alpha=transparency,
                        label='Model')
    else:
        pass

    if legend is not None:
        handles, labels = axis[1].get_legend_handles_labels()
        handlesa, labelsa = axis[0].get_legend_handles_labels()
        axis[0].legend(reversed(handlesa), reversed(labelsa),
                       markerscale=scale_of_sim_legend)
        axis[1].legend(reversed(handles), reversed(labels),
                       markerscale=scale_of_sim_legend)

    fig.tight_layout()

    if file_name is None:
        fig.show()
    else:
        fig.savefig(file_name, dpi=300)
