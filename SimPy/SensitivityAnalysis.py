from scipy.stats import spearmanr, pearsonr, rankdata
import SimPy.FormatFunctions as F
import SimPy.InOutFunctions as IO
from SimPy.Statistics import partial_corr
from collections import OrderedDict


class SensitivityAnalysis:
    def __init__(self, dic_parameter_values, output_values):
        """
        :param dic_parameter_values: (dictionary) of parameter values with parameter names as the key
        :param output_values: (list) of output values (e.g. cost or QALY observations)
        """

        self.dicParameterValues = dic_parameter_values
        self.outputValues = output_values

        for paramName, paramValues in dic_parameter_values.items():

            if len(paramValues) != len(output_values):
                raise ValueError('Number of parameter values should be equal to the number of output values. '
                                 'Error in parameter "{}".'.format(paramName))

    def _get_corr(self, f):
        """
        f: correlation function
        :returns (list) of [parameter name, correlation coefficients, p-value] """

        result = {}  # each row [parameter name, correlation, p-value]
        for paramName, paramValues in self.dicParameterValues.items():
            # calculate Spearman rank-order correlation coefficient
            coef, p = f(paramValues, self.outputValues)
            # store [parameter name, Spearman coefficient, and p-value
            result[paramName] = [coef, p]

        return result

    def get_pearson_corr(self):
        """ :returns (list) of [parameter name, Pearson's correlation coefficients, p-value] """

        return self._get_corr(f=pearsonr)

    def get_spearman_corr(self):
        """ :returns (list) of [parameter name, Spearman's rank correlation coefficients, p-value] """

        return self._get_corr(f=spearmanr)

    def get_partial_corr(self):
        """ :returns (list) of [parameter name, partial correlation coefficients, p-value] """

        result = OrderedDict()  # each row [parameter name, correlation, p-value]
        for paramName, paramValues in self.dicParameterValues.items():

            z = []
            for otherParamName, otherParamValues in self.dicParameterValues.items():
                if paramName != otherParamName:
                    z.append(otherParamValues)

            # calculate partial correlation coefficient
            coef, p = partial_corr(x=paramValues, y=self.outputValues, z=z)
            # store [parameter name, Spearman coefficient, and p-value
            result[paramName] = [coef, p]

        return result

    def get_partial_rank_corr(self):
        """ :returns (list) of [parameter name, partial rank correlation coefficients, p-value] """

        # find ranked output values
        ranked_outputs = rankdata(self.outputValues)
        # find ranked parameter values
        dic_ranked_param_values = OrderedDict()
        for paramName, paramValues in self.dicParameterValues.items():
            dic_ranked_param_values[paramName] = rankdata(paramValues)

        result = OrderedDict()  # each row [parameter name, correlation, p-value]
        for paramName, paramValues in dic_ranked_param_values.items():

            z = []
            for otherParamName, otherParamValues in dic_ranked_param_values.items():
                if paramName != otherParamName:
                    z.append(otherParamValues)

            # calculate partial correlation coefficient
            coef, p = partial_corr(x=paramValues,
                                   y=ranked_outputs,
                                   z=z)
            # store [parameter name, Spearman coefficient, and p-value
            result[paramName] = [coef, p]

        return result

    def print_corr(self, corr='r'):
        """
        :param corr: 'r' for Pearson's,
                     'rho' for Spearman's,
                     'p' for partial correlation, and
                     'pr' for partial rank correlation
        :return:
        """

        results, text = self._get_results_text(corr=corr)
        print(text)
        for par, values in results.items():
            print(par, values)

    def export_to_csv(self, corr='r', file_name='Correlations.csv', decimal=3, delimiter=','):
        """
        formats the correlation coefficients and p-value to the specified decimal point and exports to a csv file
        :param corr: 'r' for Pearson's,
                     'rho' for Spearman's,
                     'p' for partial correlation, and
                     'pr' for partial rank correlation
        :param file_name: file name
        :param decimal: decimal points to round the estimates to
        :param delimiter: to separate by comma, use ',' and by tab, use '\t'
        """
        formated_results = [['Parameter', 'Coefficient', 'P-Value']]

        results, text = self._get_results_text(corr=corr)

        for par, values in results.items():
            coef = F.format_number(number=values[0], deci=decimal)
            p = F.format_number(number=values[1], deci=decimal)
            formated_results.append([par, coef, p])

        IO.write_csv(file_name=file_name, rows=formated_results, delimiter=delimiter)

    def _get_results_text(self, corr):

        if corr == 'r':
            text = "Pearson's correlation coefficients and p-values"
            results = self.get_pearson_corr()
        elif corr == 'rho':
            text = "Spearman's rank correlation coefficients and p-values"
            results = self.get_spearman_corr()
        elif corr == 'p':
            text = "Partial correlation coefficients and p-values"
            results = self.get_partial_corr()
        elif corr == 'pr':
            text = "Partial rank correlation coefficients and p-values"
            results = self.get_partial_rank_corr()
        else:
            raise ValueError('Invalid correlation type is provided.')

        return results, text

