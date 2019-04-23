from scipy.stats import spearmanr

import SimPy.FormatFunctions as F
import SimPy.InOutFunctions as IO


class PartialRankCorrelation:
    def __init__(self, parameter_values, output_values):
        """
        :param parameter_values: (dictionary) of parameter values with parameter names as the key
        :param output_values: (list) of output values
        """

        self.results = []  # each row [parameter name, correlation, p-value]

        for paramName, paramValues in parameter_values.items():
            # calculate Spearman rank-order correlation coefficient
            coef, p = spearmanr(paramValues, output_values)
            self.results.append([paramName, coef, p])

    def export_to_csv(self, file_name='PartialRank.csv', decimal=3):

        formatedResults = [['Parameter', 'Coefficient', 'P-Value']]
        for row in self.results:
            name = row[0]
            coef = F.format_number(number=row[1], deci=decimal)
            p = F.format_number(number=row[2], deci=decimal)
            formatedResults.append([name, coef, p])

        IO.write_csv(file_name=file_name, rows=formatedResults)
