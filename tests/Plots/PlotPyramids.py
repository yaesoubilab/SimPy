import SimPy.Plots.PopulationPyramids as P

# testing
obs_data = [[0, 0, 0.1], [0, 1, 0.2], [5, 0, 0.3], [5, 1, 0.4], [10, 0, 0.6], [10, 1, 0.4]]

sim_data = [
    [[0, 0, 0.5], [0, 1, 0.3], [5, 0, 0.5], [5, 1, 0.6]],
    [[0, 0, 0.2], [0, 1, 0.4], [5, 0, 0.9], [5, 1, 0.4]]
]

P.plot_pyramids(observed_data=obs_data, simulated_data=sim_data,
                fig_size=(6, 4),
                x_lim=100,
                title='Population Pyramid in U.S.',
                colors=('blue', 'red', 'black'),
                length_of_sim_bars=750,
                scale_of_sim_legend=0.5,
                transparency=0.5
                )

age_sex_dist = [
    [8, 0, 0.055519863],   # 8, male
    [8, 1, 0.053217689],   # 8, female
    [9, 0, 0.055519863],   # 9, male
    [9, 1, 0.053217689],   # 9, female
    [10, 0, 0.056804797],  # 10, male
    [10, 1, 0.054449084],  # 10, female
    [11, 0, 0.056804798],  # 11, male
    [11, 1, 0.054449084],  # 11, female
    [12, 0, 0.056804797],  # 12, male
    [12, 1, 0.054449084],  # 12, female
    [13, 0, 0.056804797],  # 13, male
    [13, 1, 0.054449084],  # 13, female
    [14, 0, 0.056804797],  # 14, male
    [14, 1, 0.054449084],  # 14, female
    [15, 0, 0.057822037],  # 15, male
    [15, 1, 0.055305708],  # 15, female
    [16, 0, 0.057822037],  # 16, male
    [16, 1, 0.055305708]   # 16, female
]
P.plot_pyramids(observed_data=age_sex_dist,
                simulated_data=None,
                fig_size=(6, 4),
                x_lim=10,
                title='Population Pyramid',
                colors=('blue', 'red', 'black'),
                y_labels=['8', '9', '10', '11', '12', '13', '14', '15', '16'],
                age_group_width=1,
                length_of_sim_bars=750,
                scale_of_sim_legend=0.5,
                transparency=0.5
                )
