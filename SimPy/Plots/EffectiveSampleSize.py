import random

import matplotlib.pyplot as plt
import numpy as np

import SimPy.Plots.FigSupport as Fig
import SimPy.Support.MiscFunctions as S


def plot_eff_sample_size(likelihood_weights, if_randomize=True,
                         fig_size=(6, 5), file_name=None,
                         title=None, x_label='Iteration', y_label='Effective Sample Size',
                         x_range=None, y_range=None
                         ):

    # convert the data to np array if needed
    if not type(likelihood_weights) == np.ndarray:
        likelihood_weights = np.array(likelihood_weights)

    # randomize the probabilities if needed
    if if_randomize:
        random.seed(1)
        random.shuffle(likelihood_weights)

    # calculate the effectiveve sample sizes through iterations
    effs = []
    for i in range(len(likelihood_weights)):
        effs.append(S.effective_sample_size(likelihood_weights[:i + 1]))

    # plot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(range(1, len(likelihood_weights) + 1), effs)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    # if f_star is not None:
    #     plt.axhline(y=f_star, linestyle='--', color='black', linewidth=1)
    Fig.output_figure(fig, filename=file_name)

