from scipy.stats import spearmanr
import statsmodels.api as sm
import SimPy.FormatFunctions as F
import SimPy.InOutFunctions as IO


class PartialRankCorrelation:
    def __init__(self, dic_parameter_values, output_values):
        """
        :param dic_parameter_values: (dictionary) of parameter values with parameter names as the key
        :param output_values: (list) of output values (e.g. cost or QALY observations)
        """

        self.results = []  # each row [parameter name, correlation, p-value]

        for paramName, paramValues in dic_parameter_values.items():

            if len(paramValues) != len(output_values):
                raise ValueError('Number of parameter values should be equal to the number of output values. '
                                 'Error in parameter "{}".'.format(paramName))

            # calculate Spearman rank-order correlation coefficient
            coef, p = spearmanr(paramValues, output_values)

            # store [parameter name, Spearman coefficient, and p-value
            self.results.append([paramName, coef, p])

    def export_to_csv(self, file_name='PartialRank.csv', decimal=3, delimiter=','):
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


class LinearFit:

    def __init__(self, dic_parameter_values, dic_output_values):
        """
        :param dic_parameter_values: (dictionary) of parameter values with parameter names as the key
        :param dic_output_values: (dictionary) of output values (e.g. cost or QALY observations) with output
                                names as the key
        """

        assert type(dic_parameter_values) == dict, 'Parameter values should be in a dictionary'
        assert type(dic_output_values) == dict, 'Output values should be in a dictionary'

        # each row [parameter name,
        #           coeff for output 1, p-value for output 1,
        #           coeff for output 2, p=value for output 2, ...]
        self.results = []

        self.header = ['Parameter']
        for outputName in dic_output_values:
            self.header.append(outputName + ' | Coeff')
            self.header.append(outputName + ' | P-Value')

        for paramName, paramValues in dic_parameter_values.items():
            this_row = [paramName]

            for outputName, outputValues in dic_output_values.items():

                if len(paramValues) != len(outputValues):
                    raise ValueError('Number of parameter values should be equal to the number of output values. '
                                     'Error in parameter "{0}", output "{1}".'.format(paramName, outputName))

                param_values_with_constant = sm.add_constant(paramValues)

                # make a regression model
                fitted = sm.OLS(outputValues, param_values_with_constant).fit()
                #print(fitted.summary())
                if fitted.rsquared < 0.000001:
                    this_row.append(None)
                    this_row.append(None)
                else:
                    this_row.append(fitted.params[1])
                    this_row.append(fitted.pvalues[1])

            self.results.append(this_row)

    def export_to_csv(self, file_name='LinearFit.csv', decimal=3, delimiter=',', max_p_value=1):
        """
        formats the coefficients and p-value to the specified decimal point and export to a csv file
        :param file_name: file name
        :param decimal: decimal points to round the estimates to
        :param delimiter: to separate by comma, use ',' and by tab, use '\t'
        :param max_p_value: coefficients with p-value less than this will be included in the report
        """

        formatted_results = [self.header]

        for row in self.results:
            # parameter name
            this_row = [row[0]]

            n_of_outputs = (len(row)-1)/2
            for out_i in range(int(n_of_outputs)):

                p_value = row[2*out_i+2]
                if p_value is not None and p_value <= max_p_value:
                    this_row.append(F.format_number(row[2*out_i+1], deci=decimal))
                    this_row.append(F.format_number(p_value, deci=decimal))
                else:
                    this_row.append(None)
                    this_row.append(None)

            formatted_results.append(this_row)

        IO.write_csv(file_name=file_name, rows=formatted_results, delimiter=delimiter)