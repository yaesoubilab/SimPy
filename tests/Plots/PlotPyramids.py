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
