from SimPy.SensitivityAnalysis import SensitivityAnalysis


def do_sa(dic_parameter_values, outputs):
    sa = SensitivityAnalysis(dic_parameter_values=dic_parameter_values,
                             output_values=outputs)

    print('')
    sa.print_corr(corr='r')     # Pearson's
    sa.print_corr(corr='rho')   # Spearman's
    sa.print_corr(corr='p')     # partial correlation
    sa.print_corr(corr='pr')    # partial rank correlation


dic_parameter_values = {
    'par1': [1, 4, 3, 7, 1],
    'par2': [5, 7, 2, 8, 23],
    'par3': [5, 6, 7, 8, 9]
}
outputs = [7, 1, 2, 15, 5]

do_sa(dic_parameter_values=dic_parameter_values, outputs=outputs)


dic_parameter_values = {
    'par1': [1, 1, 1, 1, 1],
    'par2': [5, 7, 2, 8, 23],
}
outputs = [7, 1, 2, 15, 5]

do_sa(dic_parameter_values=dic_parameter_values, outputs=outputs)
