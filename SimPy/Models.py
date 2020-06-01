from SimPy.DataFrames import DataFrameOfObjects
import SimPy.RandomVariateGenerators as RVGs


class MortalityModel:
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31

    This class assumes that the first column contains age groups and the last column contains mortality rates
    """

    def __init__(self, rows, group_mins, group_maxs, group_delta, age_min, age_delta):
        """
        :param rows: (list of list) the table above
        :param group_mins: list of minimum value of x (in example above: [0, 0])
        :param group_maxs: list of maximum value of x (in example above: [10, 1])
        :param group_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        :param age_min:
        :param age_delta:
        """

        self.df = DataFrameOfObjects(list_x_min=group_mins,
                                     list_x_max=group_maxs,
                                     list_x_delta=group_delta)

        self.ageMin = age_min

        for df_row in self.df.get_rows():
            rates = []
            for row in rows:
                if df_row[:-1] == row[1:-1]:
                    rates.append(row[-1])
            self.df.set_obj(x_value=df_row[0:-1],
                            obj=RVGs.NonHomogeneousExponential(rates=rates, delta_t=age_delta))

    def sample_time_to_death(self, group, age, rng):

        if age < self.ageMin:
            raise ValueError('Current age cannot be smaller than the minimum age.')

        return self.df.get_obj(group).sample(rng=rng, arg=age)
