from SimPy.DataFrames import Pyramid
import numpy as np
import matplotlib.pyplot as plt


def plot_pyramids(observed_pyramid, simulated_pyramids,
                  x_lim, y_lim, title,
                  y_ticks_places=None, y_labels=None,
                  fig_size=None, age_group_length=5):
    """
    :param observed_pyramid:
    :param simulated_pyramids:
    :param x_lim:
    :param y_lim:
    :param title:
    :param y_ticks_places:
    :param y_labels:
    :param fig_size: (tuple) figure size (e.g. (2, 4))
    :param age_group_length: length of each age group in years.
    :return:
    """

    # women_count/ men_count = array counter
    women_count = 0
    men_count = 0
    # men and women percent of population array
    womens_list = []
    mens_list = []
    w_age = []
    m_age = []

    # m_num_age/ w_num_age = keep track of the age of the men and women
    length = len(observed_pyramid)
    i = 0
    # sort data into separate arrays
    while i < length:
        # woman
        if observed_pyramid[i][1] == 1:
            womens_list.insert(women_count, observed_pyramid[i][2] * 100)
            w_age.insert(women_count, observed_pyramid[i][0])
            women_count = women_count + 1
            i = i + 1
        # men
        else:
            mens_list.insert(men_count, observed_pyramid[i][2] * 100)
            m_age.insert(men_count, observed_pyramid[i][0])
            men_count = men_count + 1
            i = i + 1

    # plot
    fig, axis = plt.subplots(ncols=2, sharey=True, tight_layout=True, figsize=fig_size)
    st = fig.suptitle(title)

    # either sets to default ticks or user input
    if y_ticks_places is None:
        y_ticks_places = range(0, 100, age_group_length)

    # either sets to default tick labels or user input
    if y_labels is None:
        y_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44',
                    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
                    '80-89', '90-94', '95-100', '100+']

    # plot configuration
    plt.setp(axis, yticks=y_ticks_places, yticklabels=y_labels)
    axis[0].xmin, axis[0].xmax, axis[0].ymin, axis[0].ymax \
        = axis[0].axis([0, x_lim, -age_group_length/2-0.5, y_lim+0.5])
    axis[1].xmin, axis[1].xmax, axis[1].ymin, axis[1].ymax \
        = axis[1].axis([0, x_lim, -age_group_length/2-0.5, y_lim+0.5])
    axis[0].barh(m_age, mens_list, age_group_length, align='center', color='brown', edgecolor='black', alpha=0.4)
    axis[0].invert_xaxis()
    axis[0].set(title='\nMen')
    axis[1].barh(w_age, womens_list, age_group_length, align='center', color='gray', edgecolor='black', alpha=0.4)
    axis[1].set(title='\nWomen')
    axis[0].set_ylabel('Age')
    axis[0].set_xlabel('Percent of Population')
    axis[1].set_xlabel('Percent of Population')

    # adding pyramids from simulation
    if simulated_pyramids is not None:
        num_length = len(simulated_pyramids)
        j = 0
        k = 0
        while j < num_length:
            while k < len(simulated_pyramids[j]):
                if simulated_pyramids[j][k][1] == 1:
                    axis[0].plot(simulated_pyramids[j][k][2] * 100, simulated_pyramids[j][k][0],
                                 marker='|', mec="blue",
                                 markersize=10*age_group_length/2, color="blue", alpha=0.6)
                    k = k + 1
                else:
                    axis[1].plot(simulated_pyramids[j][k][2] * 100, simulated_pyramids[j][k][0],
                                 marker='|', mec="blue",
                                 markersize=10*age_group_length/2, color="blue", alpha=0.6)
                    k = k + 1
            k = 0
            j = j + 1
    else:
        pass

    fig.tight_layout()
    # st.set_y(1)
    # fig.subplots_adjust(top=1)
    fig.show()


# testing
# data_table = [[0, 0, 0.1], [0, 1, 0.2], [5, 0, 0.3], [5, 1, 0.4], [10, 0, 0.6], [10, 1, 0.4]]
# sim_table = [
#     [[0, 0, 0.5], [0, 1, 0.3], [5, 0, 0.5], [5, 1, 0.6]],
#     [[0, 0, 0.2], [0, 1, 0.4], [5, 0, 0.9], [5, 1, 0.4]]
# ]
#
# pyrfig = plot_pyramids(data_list=data_table, sim_list= sim_table, fig_size=(6, 4),
#                       x_lim=100, y_lim=12.5,
#                       title='Population Pyramid in U.S.')
# pyrfig.show()
# pyrfig.savefig("pyramid.png")