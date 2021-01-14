from SimPy.SensitivityAnalysis import SensitivityAnalysis


dic_parameter_values = {
    'par1': [1, 4, 3, 7, 1],
    'par2': [5, 7, 2, 8, 23],
    'par3': [5, 6, 7, 8, 9]
}
outputs = [7, 1, 2, 15, 5]


sa = SensitivityAnalysis(dic_parameter_values=dic_parameter_values,
                         output_values=outputs)

sa.print_corr(corr='r')
sa.print_corr(corr='rho')
sa.print_corr(corr='p')
sa.print_corr(corr='pr')
