import SimPy.InOutFunctions as IO
import SimPy.Plots.Histogram as Fig
import SimPy.StatisticalClasses as Stat
import SimPy.FormatFunctions as Format
import SimPy.Support.MiscFunctions as F
import copy
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import pearsonr
from numpy.random import choice


# list of columns in the parameter csv file that are not considered a parameter
COLUMNS_TO_SKIP = ['ID', 'Seed', 'Likelihood Weights', 'Simulation Replication', 'Random Seed']
HISTOGRAM_FIG_SIZE = (4.2, 3.2)


class ColumnsPriorDistCSV(Enum):
    # columns of the csv file containing parameter prior distributions
    ID = 0
    NAME = 1
    LB = 2
    UB = 3
    TITLE = 4
    MULTIPLIER = 5
    FORMAT = 6
    DECI = 7


class ParameterSampler:
    # to sample parameter values according to their likelihood weights
    def __init__(self, csv_file_name):

        self.rowsOfParameters = IO.read_csv_rows(file_name=csv_file_name,
                                                 if_ignore_first_row=False,
                                                 if_convert_float=True)

    def select_param_values(self, n, weight_col, filename, sample_by_weight=True, seed=0):

        # header
        rows = [self.rowsOfParameters[0]]

        if not sample_by_weight:
            for i, row in enumerate(self.rowsOfParameters):
                if i > 0:
                    if row[weight_col] > 0 and i <= n:
                        rows.append(row)
        else:
            # weight
            weights = []
            for row in self.rowsOfParameters:
                weights.append(row[weight_col])
            del(weights[0])

            # sample rows
            rng = np.random.RandomState(seed=seed)
            sampled_indices = rng.choice(a=range(len(weights)), size=n, p=weights)

            # build sampled rows
            for i in sampled_indices:
                rows.append(self.rowsOfParameters[i+1])

        IO.write_csv(rows=rows, file_name=filename)


class ParamInfo:
    # class to store information about a parameter (id, name, estimate and confidence/uncertainty interval)
    def __init__(self, idx, name, label=None, values=None, range=None):
        self.idx = idx
        self.name = name
        self.label = label
        self.values = np.array(values)
        self.range = range
        self.estimate = None
        self.interval = None


class ParameterAnalyzer:
    # class to create a dictionary of parameters
    def __init__(self, csv_file_name):
        """
        :param csv_file_name: csv file where the parameter samples are located
        assumes that the first row of this csv file contains the parameter names
        to be used as the keys of the dictionary of parameters it creates
        """

        # create a dictionary of parameter samples
        self.dictOfParams = IO.read_csv_cols_to_dictionary(file_name=csv_file_name,
                                                           if_convert_float=True)

    def get_mean_interval(self, parameter_name, deci=0, form=None):

        # print the ratio estimate and credible interval
        sum_stat = Stat.SummaryStat('', self.dictOfParams[parameter_name])

        if form is None:
            return sum_stat.get_mean(), sum_stat.get_PI(alpha=0.05)
        else:
            return sum_stat.get_formatted_mean_and_interval(
                interval_type='p', alpha=0.05, deci=deci, form=form)

    def plot_histogram(self, parameter_name, title, x_label=None, y_label=None, x_range=None, folder=''):
        """ creates a histogram of one parameter """

        Fig.plot_histogram(
            data=self.dictOfParams[parameter_name],
            title=title, x_label=x_label, y_label=y_label,
            x_range=x_range, figure_size=HISTOGRAM_FIG_SIZE, file_name=folder+'/'+title
        )

    def plot_histograms(self, ids=None, names=None, csv_file_name_prior=None, posterior_fig_loc='figures_national'):
        """ creates histograms of parameters specified by ids
        :param ids: (list) of parameter ids to plot
        :param names: (list) of parameter names to plot
        :param csv_file_name_prior: (string) filename where parameter prior ranges are located
        :param posterior_fig_loc: (string) location where posterior figures_national should be located
        """

        # clean the directory
        IO.delete_files('.png', posterior_fig_loc)

        # read prior distributions
        if csv_file_name_prior is not None:
            priors = IO.read_csv_rows(
                file_name=csv_file_name_prior,
                if_ignore_first_row=True,
                delimiter=',',
                if_convert_float=True
            )

        # for all parameters, read sampled parameter values and create the histogram
        par_id = 0
        for par_name, par_values in self.dictOfParams.items():

            # skip the columns that are not considered parameter
            if par_name in COLUMNS_TO_SKIP:
                continue

            # check if the histogram should be created for this parameter
            if_show = self._if_include(par_id=par_id, par_name=par_name, ids=ids, names=names)

            # create the histogram
            if if_show:
                # find prior range
                x_range = None
                if priors is not None:
                    try:
                        x_range = [float(priors[par_id][ColumnsPriorDistCSV.LB.value]), float(priors[par_id][ColumnsPriorDistCSV.UB.value])]
                    except:
                        print('Could not convert string to float to find the prior distribution of parameter:', par_id)
                else:
                    x_range = None

                # find the filename the histogram should be saved as
                file_name = posterior_fig_loc + '\Par-' + str(par_id) + ' ' + F.proper_file_name(par_name)

                # find title
                if priors[par_id][ColumnsPriorDistCSV.TITLE.value] in ('', None):
                    title = priors[par_id][ColumnsPriorDistCSV.NAME.value]
                else:
                    title = priors[par_id][ColumnsPriorDistCSV.TITLE.value]

                # find multiplier
                if priors[par_id][ColumnsPriorDistCSV.MULTIPLIER.value] in ('', None):
                    multiplier = 1
                else:
                    multiplier = float(priors[par_id][ColumnsPriorDistCSV.MULTIPLIER.value])
                x_range = [x*multiplier for x in x_range]
                par_values = [v*multiplier for v in par_values]

                # plot histogram
                Fig.plot_histogram(
                    data=par_values,
                    title=title.replace('!', '\n'),
                    x_range=x_range,
                    figure_size=HISTOGRAM_FIG_SIZE,
                    file_name=file_name
                )

            # move to the next parameter
            par_id += 1

    def plot_pairwise(self, ids=None, names=None, csv_file_name_prior=None, fig_filename='pairwise_correlation.png',
                      figure_size=(10, 10)):
        """ creates pairwise corrolation between parameters specified by ids
        :param ids: (list) ids of parameters to display
        :param names: (list) names of parameter to display
        :param csv_file_name_prior: (string) filename where parameter prior ranges are located
        :param fig_filename: (string) filename to save the figure as
        :param figure_size: (tuple) figure size
        """

        # read prior distributions
        priors = None
        if csv_file_name_prior is not None:
            priors = IO.read_csv_rows(
                file_name=csv_file_name_prior,
                if_ignore_first_row=True,
                delimiter=',',
                if_convert_float=True
            )

        # find the names of parameters to include in the analysis
        info_of_params_to_include = []

        par_id = 0
        for par_name, par_values in self.dictOfParams.items():

            # skip these columns
            if par_name in COLUMNS_TO_SKIP:
                continue

            # check if the histogram should be created for this parameter
            if_show = self._if_include(par_id=par_id, par_name=par_name, ids=ids, names=names)

            # create the histogram
            if if_show:
                # find prior range
                x_range = None
                if priors is not None:
                    try:
                        x_range = [float(priors[par_id][ColumnsPriorDistCSV.LB.value]), float(priors[par_id][ColumnsPriorDistCSV.UB.value])]
                    except:
                        print('Could not convert string to float to find the prior distribution of parameter:', par_id)
                else:
                    x_range = None

                # find title
                if priors is not None:
                    if  priors[par_id][ColumnsPriorDistCSV.TITLE.value] in ('', None):
                        label = priors[par_id][ColumnsPriorDistCSV.NAME.value]
                    else:
                        label = priors[par_id][ColumnsPriorDistCSV.TITLE.value]
                else:
                    label = par_name

                # find multiplier
                if priors is not None:
                    if priors[par_id][ColumnsPriorDistCSV.MULTIPLIER.value] in ('', None):
                        multiplier = 1
                    else:
                        multiplier = float(priors[par_id][ColumnsPriorDistCSV.MULTIPLIER.value])
                else:
                    multiplier = 1

                if x_range is not None:
                    x_range = [x*multiplier for x in x_range]
                par_values = [v*multiplier for v in par_values]

                # append the info for this parameter
                info_of_params_to_include.append(
                    ParamInfo(idx=par_id, name=par_name, label=label.replace('!', '\n'),
                              values=par_values, range=x_range)
                )

            # move to the next parameter
            par_id += 1

        # plot pairwise
        # set default properties of the figure
        plt.rc('font', size=6)  # fontsize of texts
        plt.rc('axes', titlesize=6)  # fontsize of the figure title
        plt.rc('axes', titleweight='semibold')  # fontweight of the figure title

        # plot each panel
        n = len(info_of_params_to_include)
        f, axarr = plt.subplots(nrows=n, ncols=n, figsize=figure_size)

        for i in range(n):
            for j in range(n):

                # get the current axis
                ax = axarr[i, j]

                if j == 0:
                    ax.set_ylabel(info_of_params_to_include[i].label)
                if i == n-1:
                    ax.set_xlabel(info_of_params_to_include[j].label)

                if i == j:
                    # plot histogram
                    Fig.add_histogram_to_ax(
                        ax=ax,
                        data=info_of_params_to_include[i].values,
                        x_range=info_of_params_to_include[i].range
                    )
                    ax.set_yticklabels([])
                    ax.set_yticks([])

                else:
                    ax.scatter(info_of_params_to_include[j].values,
                               info_of_params_to_include[i].values,
                               alpha=0.5, s=2)
                    ax.set_xlim(info_of_params_to_include[j].range)
                    ax.set_ylim(info_of_params_to_include[i].range)
                    # correlation line
                    b, m = polyfit(info_of_params_to_include[j].values,
                                   info_of_params_to_include[i].values, 1)
                    ax.plot(info_of_params_to_include[j].values,
                            b + m * info_of_params_to_include[j].values, '-', c='black')
                    corr, p = pearsonr(info_of_params_to_include[j].values,
                                       info_of_params_to_include[i].values)
                    ax.text(0.95, 0.95, '{0:.2f}'.format(corr), transform=ax.transAxes, fontsize=6,
                            va='top', ha='right')

        f.align_ylabels(axarr[:, 0])
        f.tight_layout()
        f.savefig(fig_filename, bbox_inches='tight', dpi=300)

    def export_means_and_intervals(self,
                                   poster_file='ParameterEstimates.csv',
                                   significance_level=0.05, sig_digits=3,
                                   ids=None, names=None, prior_info_csv_file=None):
        """ calculate the mean and credible intervals of parameters specified by ids
        :param poster_file: csv file where the posterior ranges should be stored
        :param significance_level:
        :param sig_digits: number of significant digits
        :param ids:
        :param names: 
        :param prior_info_csv_file: (string) filename where parameter prior ranges are located
        :return:
        """

        results = self.get_means_and_intervals(significance_level=significance_level,
                                               sig_digits=sig_digits,
                                               ids=ids, names=names,
                                               prior_info_csv_file=prior_info_csv_file)

        # write parameter estimates and credible intervals
        IO.write_csv(rows=results, file_name=poster_file)

    def print_means_and_intervals(self,
                                  significance_level=0.05, sig_digits=3,
                                  ids=None, names=None, prior_info_csv_file=None):
        """ calculate the mean and credible intervals of parameters specified by ids
        :param significance_level:
        :param sig_digits: number of significant digits
        :param ids:
        :param names:
        :param prior_info_csv_file: (string) filename where parameter prior ranges are located
        :return:
        """

        results = self.get_means_and_intervals(significance_level=significance_level,
                                               sig_digits=sig_digits,
                                               ids=ids, names=names,
                                               prior_info_csv_file=prior_info_csv_file)
        for r in results:
            print(r)

    def get_means_and_intervals(self, significance_level=0.05, sig_digits=3,
                                ids=None, names=None, prior_info_csv_file=None):
        # read prior distributions
        priors = None
        if prior_info_csv_file is not None:
            priors = IO.read_csv_rows(
                file_name=prior_info_csv_file,
                if_ignore_first_row=True,
                delimiter=',',
                if_convert_float=True
            )

        results = []  # list of parameter estimates and credible intervals

        par_id = 0
        for par_name, par_values in self.dictOfParams.items():

            # skip these columns
            if par_name in COLUMNS_TO_SKIP:
                continue

            # if estimates and credible intervals should be calculated for this parameter
            if_record = self._if_include(par_id=par_id, par_name=par_name, ids=ids, names=names)

            # record the calculated estimate and credible interval
            if if_record:

                if priors is None:
                    decimal = None
                    form = ''
                    multip = 1
                else:
                    decimal = priors[par_id][ColumnsPriorDistCSV.DECI.value]
                    decimal = 0 if decimal is None else decimal
                    sig_digits = None
                    form = priors[par_id][ColumnsPriorDistCSV.FORMAT.value]
                    multip = priors[par_id][ColumnsPriorDistCSV.MULTIPLIER.value]

                if multip is None:
                    data = par_values
                else:
                    multip = float(multip)
                    data = [multip * x for x in par_values]

                sum_stat = Stat.SummaryStat(name=par_name, data=data)
                mean_text = Format.format_number(number=sum_stat.get_mean(),
                                                 deci=decimal, sig_digits=sig_digits, format=form)
                PI_text = Format.format_interval(interval=sum_stat.get_PI(significance_level),
                                                 deci=decimal, sig_digits=sig_digits, format=form)

                results.append(
                    [par_id, par_name, mean_text, PI_text]
                )

            # next parameter
            par_id += 1

        return results

    def __calculate_ratio_obss(self, numerator_par_name, denominator_par_names):

        # if only one parameter is in the denominator
        if type(denominator_par_names) is not list:
            denominator_par_names = [denominator_par_names]

        # calculate sum of parameters in the denominator
        sum_denom = copy.deepcopy(self.dictOfParams[denominator_par_names[0]])
        for i in range(1, len(denominator_par_names)):
            sum_denom += self.dictOfParams[denominator_par_names[i]]

        # calculate realizations for ratio
        return self.dictOfParams[numerator_par_name]/sum_denom

    @staticmethod
    def _if_include(par_id, par_name, ids=None, names=None):
        # check if this parameter should be included
        if_include = False
        if ids is None and names is None:
            if_include = True
        else:
            if ids is not None and par_id in ids:
                if_include = True
            if names is not None and par_name in names:
                if_include = True
        return if_include

    def get_ratio_mean_interval(self, numerator_par_name, denominator_par_names, deci=0, form=None):

        # print the ratio estimate and credible interval
        sum_stat = Stat.SummaryStat('', self.__calculate_ratio_obss(numerator_par_name, denominator_par_names))

        if form is None:
            return sum_stat.get_mean(), sum_stat.get_PI(alpha=0.05)
        else:
            return sum_stat.get_formatted_mean_and_interval(
                interval_type='p', alpha=0.05, deci=deci, form=form)

    def plot_ratio_hist(self, numerator_par_name, denominator_par_names,
                        title, x_label=None, x_range=None, output_fig_loc='figures_national'):

        ratio_obss = self.__calculate_ratio_obss(numerator_par_name, denominator_par_names)

        file_name = output_fig_loc + '\Ratio-' + title

        # create the histogram of ratio
        Fig.plot_histogram(
            data=ratio_obss,
            title=title,
            x_label=x_label,
            x_range=x_range,
            figure_size=HISTOGRAM_FIG_SIZE,
            output_type='png',
            file_name=file_name)