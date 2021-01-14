from scipy.stats import spearmanr, pearsonr, rankdata
import statsmodels.api as sm
import SimPy.FormatFunctions as F
import SimPy.InOutFunctions as IO
from SimPy.StatisticalClasses import partial_corr


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

        result = {}  # each row [parameter name, correlation, p-value]
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

        result = {}  # each row [parameter name, correlation, p-value]
        for paramName, paramValues in self.dicParameterValues.items():

            z = []
            for otherParamName, otherParamValues in self.dicParameterValues.items():
                if paramName != otherParamName:
                    z.append(rankdata(otherParamValues))

            # calculate partial correlation coefficient
            coef, p = partial_corr(x=rankdata(paramValues),
                                   y=rankdata(self.outputValues),
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
        if corr == 'r':
            print("Pearson's correlation coefficients and p-values")
            results = self.get_pearson_corr()
        elif corr == 'rho':
            print("Spearman's rank correlation coefficients and p-values")
            results = self.get_spearman_corr()
        elif corr == 'p':
            print("Partial correlation coefficients and p-values")
            results = self.get_partial_corr()
        elif corr == 'pr':
            print("Partial rank correlation coefficients and p-values")
            results = self.get_partial_rank_corr()
        else:
            raise ValueError('Invalid correlation type is provided.')

        for par, values in results.items():
            print(par, values)

    def export_to_csv(self, file_name='Correlations.csv', decimal=3, delimiter=','):
        """
        formats the coefficients and p-value to the specified decimal point and export to a csv file
        :param file_name: file name
        :param decimal: decimal points to round the estimates to
        :param delimiter: to separate by comma, use ',' and by tab, use '\t'
        """
        formated_results = [['Parameter', 'Coefficient', 'P-Value']]

        for row in self.results:
            name = row[0]
            coef = F.format_number(number=row[1], deci=decimal)
            p = F.format_number(number=row[2], deci=decimal)
            formated_results.append([name, coef, p])

        IO.write_csv(file_name=file_name, rows=formated_results, delimiter=delimiter)


class ParameterSA:

    def __init__(self, dic_parameter_values, dic_output_values):
        """
        :param dic_parameter_values: (dictionary) of parameter values with parameter names as the key
        :param dic_output_values: (dictionary) of output values (e.g. cost or QALY observations) with output
                                names as the key
        """

        assert isinstance(dic_parameter_values, dict), \
            'Parameter values should be a dictionary.'
        assert isinstance(dic_output_values, dict), \
            'Output values should be a dictionary.'

        self.results_linear_fit = []  # for linear fit
        # each row [parameter name,
        #           coeff for output 1, p-value for output 1,
        #           coeff for output 2, p=value for output 2, ...]

        self.results_prcc = []  # for partial rank correlation
        # each row [parameter name,
        #           correlation for output 1, p-value for output 1,
        #           correlation for output 2, p=value for output 2, ...]

        # make the header if need to export to cvs files
        self.header = ['Parameter']
        for outputName in dic_output_values:
            self.header.append(outputName + ' | Coeff')
            self.header.append(outputName + ' | P-Value')

        # calculate linear fit and prcc
        for paramName, paramValues in dic_parameter_values.items():
            this_row_linear_fit = [paramName]
            this_row_prc = [paramName]

            for outputName, outputValues in dic_output_values.items():

                if len(paramValues) != len(outputValues):
                    raise ValueError('Number of parameter values should be equal to the number of output values. '
                                     'Error in parameter "{0}", output "{1}".'.format(paramName, outputName))

                # make a regression model
                param_values_with_constant = sm.add_constant(paramValues)
                fitted = sm.OLS(outputValues, param_values_with_constant).fit()
                if fitted.rsquared < 0.000001:
                    this_row_linear_fit.append(None)
                    this_row_linear_fit.append(None)
                else:
                    this_row_linear_fit.append(fitted.params[1])
                    this_row_linear_fit.append(fitted.pvalues[1])

                # calculate Spearman rank-order correlation coefficient
                coef, p = spearmanr(paramValues, outputValues)
                this_row_prc.append(coef)
                this_row_prc.append(p)

            self.results_linear_fit.append(this_row_linear_fit)
            self.results_prcc.append(this_row_prc)

    def export_to_csv(self, file_name_linear_fit='LinearFit.csv', file_name_prcc='PRCC.csv',
                      decimal=3, delimiter=',', max_p_value=1):
        """
        formats the coefficients and p-value to the specified decimal point and export to a csv file
        :param file_name_linear_fit: file name to store the results of linear fit
        :param file_name_prcc: file name to store the results of PRCC
        :param decimal: decimal points to round the estimates to
        :param delimiter: to separate by comma, use ',' and by tab, use '\t'
        :param max_p_value: coefficients with p-value less than this will be included in the report
        """

        formatted_results_linear_fit = [self.header]
        formatted_results_prcc = [self.header]

        n_of_outputs = (len(self.results_linear_fit[0]) - 1) / 2
        n_of_params = len(self.results_linear_fit)

        # format the results of linear fit analysis
        for i in range(n_of_params):
            # parameter name
            this_row_linear_fit = [self.results_linear_fit[i][0]]
            this_row_prcc = [self.results_prcc[i][0]]

            for j in range(int(n_of_outputs)):

                coeff_linear_fit = self.results_linear_fit[i][2*j+1]
                p_value_linear_fit = self.results_linear_fit[i][2*j+2]
                coeff_prcc = self.results_prcc[i][2*j+1]
                p_value_prcc = self.results_prcc[i][2*j+2]

                # format results of linear fit
                if p_value_linear_fit is not None and p_value_linear_fit <= max_p_value:
                    this_row_linear_fit.append(F.format_number(coeff_linear_fit, deci=decimal))
                    this_row_linear_fit.append(F.format_number(p_value_linear_fit, deci=decimal))
                else:
                    this_row_linear_fit.append(None)
                    this_row_linear_fit.append(None)

                # format results of linear prcc
                if p_value_prcc is not None and p_value_prcc <= max_p_value:
                    this_row_prcc.append(F.format_number(coeff_prcc, deci=decimal))
                    this_row_prcc.append(F.format_number(p_value_prcc, deci=decimal))
                else:
                    this_row_prcc.append(None)
                    this_row_prcc.append(None)

            formatted_results_linear_fit.append(this_row_linear_fit)
            formatted_results_prcc.append(this_row_prcc)

        IO.write_csv(file_name=file_name_linear_fit, rows=formatted_results_linear_fit, delimiter=delimiter)
        IO.write_csv(file_name=file_name_prcc, rows=formatted_results_prcc, delimiter=delimiter)



