import matplotlib.pyplot as plt


def plot_pyramids(observed_data, simulated_data,
                  title=None, fig_size=None,
                  x_lim=100,
                  y_labels=None,
                  age_group_length=5,
                  colors=('brown', 'grey'),
                  transparency=0.4):
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
    :param age_group_length: length of each age group in years.
    :param colors: (tuple) example: ('brown', 'grey')
    :param transparency: transparency of bars
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
    y_lim = m_ages[-1] + age_group_length/2

    # either sets to default ticks or user input
    y_ticks_places = range(0, len(m_ages) * age_group_length, age_group_length)

    # either sets to default tick labels or user input
    if y_labels is None:
        y_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
                    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
                    '80-89', '90-94', '95-100', '100+']

    # plot configuration
    fig, axis = plt.subplots(ncols=2, sharey=True, tight_layout=True, figsize=fig_size)
    # figure title
    st = fig.suptitle(title)
    # add y labels and ticks
    plt.setp(axis, yticks=y_ticks_places, yticklabels=y_labels)

    # ranges of x- and y-axes
    for i in [0, 1]:
        axis[i].xmin, axis[i].xmax, axis[i].ymin, axis[i].ymax \
            = axis[i].axis([0, x_lim, -age_group_length*(1/2+1/10), y_lim+age_group_length*1/10])

    # bar plot for males
    axis[0].barh(m_ages, m_sizes, age_group_length,
                 align='center', color=colors[0], edgecolor='black', alpha=transparency, label='Data')
    axis[0].invert_xaxis()

    # bar plot for females
    axis[1].barh(w_ages, w_sizes, age_group_length,
                 align='center', color=colors[1], edgecolor='black', alpha=transparency, label='Data')

    # title of each sub-figure
    axis[0].set(title='\nMen')
    axis[1].set(title='\nWomen')

    # add labels
    axis[0].set_ylabel('Age')
    axis[0].set_xlabel('Percent of Population')
    axis[1].set_xlabel('Percent of Population')

    # adding pyramids from simulation
    if simulated_data is not None:
        num_length = len(simulated_data)
        j = 0
        k = 0
        while j < num_length:
            while k < len(simulated_data[j]):
                if simulated_data[j][k][1] == 1:
                    axis[0].plot(simulated_data[j][k][2] * 100, simulated_data[j][k][0],
                                 marker='|', mec="blue",
                                 markersize=10*age_group_length/2,
                                 mew=1.5,     # increase this for ticker bars
                                 color="blue", alpha=transparency)
                    k = k + 1
                else:
                    axis[1].plot(simulated_data[j][k][2] * 100, simulated_data[j][k][0],
                                 marker='|', mec="blue",
                                 markersize=10*age_group_length/2,
                                 mew=1.5,  # increase this for ticker bars
                                 color="blue", alpha=transparency)
                    k = k + 1
            k = 0
            j = j + 1
    else:
        pass

    fig.tight_layout()
    # st.set_y(1)
    # fig.subplots_adjust(top=1)
    fig.show()
