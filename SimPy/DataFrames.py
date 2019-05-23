import SimPy.RandomVariantGenerators as RVGs
import numpy as np


class OneDimDataFrame:
    """
    example:
        age,    mortality rate
        0,      0.1,
        5,      0.2,
        10,     0.3
    """
    def __init__(self, y_objects, x_min=0, x_max=1, x_delta=1):
        """
        :param y_objects: (list) of y objects (in example above: [0.1, 0.2, 0.3])
                but it could also be list of other objects of the same type.
        :param x_min: minimum value of x (in example above: 0)
        :param x_max: maximum value of x (in example above: 10)
        :param x_delta: interval between break points of x (in example above: 5)
                    if set to 'int', x is treated as categorical variable
        """

        self.objs = y_objects
        self.xDelta = x_delta
        self.xMin = x_min
        self.xMax = x_max

    def get_index(self, x_value):
        """ :returns the index of the smallest x break point with x_value greater than or equal.
            In the example above, it returns
                0 for 0 <= x_value < 5,
                1 for 5 <= x_value < 10, and
                2 for x_value >= 10
        """

        if self.xDelta == 'int':
            if type(x_value) is not int:
                raise ValueError('x_value should be an integer for categorical variables.')
            return x_value
        else:
            if x_value >= self.xMax:
                return len(self.objs) - 1
            else:
                return round((x_value-self.xMin)/self.xDelta)

    def get_obj(self, x_value):
        """ :returns the the object in the smallest x break point with x_value greater than or equal.
            In the example above, it returns
                0.1 for 0 <= x_value < 5,
                0.2 for 5 <= x_value < 10, and
                0.3 for x_value >= 10
        """

        return self.objs[self.get_index(x_value)]

    def get_obj_by_index(self, x_index):
        """ :returns the the object in the x break point located at index x_index.
            In the example above, it returns
                0.1 for x_index = 0,
                0.2 for x_index = 1, and
                0.3 for x_index = 2
        """

        return self.objs[x_index]


class OneDimDataFrameWithExpDist (OneDimDataFrame):
    """
    example:
        age,    mortality rate
        0,      0.1,
        5,      0.2,
        10,     0.3
    """
    def __init__(self, y_values, x_min=0, x_max=1, x_delta=1):
        """
        :param y_values: (list) of rates (in example above: [0.1, 0.2, 0.3]
        :param x_min: minimum value of x (in example above: 0)
        :param x_max: maximum value of x (in example above: 10)
        :param x_delta: interval between break points of x (in example above: 5)
                    if set to 'int', x is treated as categorical variable
        """

        y_objects = []
        for v in y_values:
            y_objects.append(RVGs.Exponential(scale=1/v))

        OneDimDataFrame.__init__(self, y_objects=y_objects,
                                 x_min=x_min,
                                 x_max=x_max,
                                 x_delta=x_delta)

    def get_dist(self, x_value):
        """ :returns the the exponential distribution in the smallest x break point
            with x_value greater than or equal.
            In the example above, it returns
                exp(0.1) for 0 <= x_value < 5,
                exp(0.2) for 5 <= x_value < 10, and
                exp(0.3) for x_value >= 10
        """
        return self.get_obj(x_value)

    def get_dist_by_index(self, x_index):
        """ :returns the the exponential distribution in the x break point located at index x_index.
            In the example above, it returns
                exp(0.1) for x_index = 0,
                exp(0.2) for x_index = 1, and
                exp(0.3) for x_index = 2
        """
        return self.get_obj_by_index(x_index)


class _DataFrame:
    def __init__(self, list_x_min, list_x_max, list_x_delta):

        if not isinstance(list_x_min, list):
            list_x_min = [list_x_min]
        if not isinstance(list_x_max, list):
            list_x_max = [list_x_max]
        if not isinstance(list_x_delta, list):
            list_x_delta = [list_x_delta]

        self._xMin = list_x_min[0]
        self._xMax = list_x_max[0]
        self._xDelta = list_x_delta[0]

        self._xs = []           # x breakpoints
        self._ifOneDim = False
        self._ifContinuous = True
        self._objs = []      # objects of this data frame
        self._dataFrames = []

        if len(list_x_min) == 1:
            self._ifOneDim = True
        if list_x_delta[0] == 'int':
            self._ifContinuous = False

        # find x breakpoints
        x = list_x_min[0]
        while x <= list_x_max[0]:
            self._xs.append(x)
            if self._ifContinuous:
                x += list_x_delta[0]
            else:
                x += 1

        # build data frames
        for x in self._xs:
            if self._ifOneDim:
                self._objs.append(0)
            else:
                self._dataFrames.append(
                    _DataFrame(list_x_min=list_x_min[1:],
                               list_x_max=list_x_max[1:],
                               list_x_delta=list_x_delta[1:]
                               )
                )

    def __get_index(self, x_value):

        if not isinstance(x_value, list):
            x_value = [x_value]

        if self._xDelta == 'int':
            if type(x_value[0]) is not int:
                raise ValueError('x_value should be an integer for categorical variables.')
            return x_value[0] - self._xMin
        else:
            if x_value[0] > self._xMax:
                return len(self._objs) - 1
            else:
                return round((x_value[0] - self._xMin) / self._xDelta)

    def get_sum(self):
        if self._ifOneDim:
            return sum(self._objs)
        else:
            sum_df = 0
            for df in self._dataFrames:
                sum_df += df.get_sum()
            return sum_df

    def get_percentage(self):
        total = self.get_sum()
        rows = []
        for row in self.get_rows():
            a = row[:-1]
            a.extend([row[-1]/total])
            rows.append(a)

        return rows

    def update_value(self, x_value, v):

        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            self._objs[index] = v
        else:
            self._dataFrames[index].update_value(x_value=x_value[1:], v=v)

    def get_index(self, x_value):

        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            return [index]
        else:
            return self._dataFrames[index].get_index(x_value=x_value[1:])

    def get_obj(self, x_value):

        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            return self._objs[index]
        else:
            return self._dataFrames[index].get_obj(x_value[1:])

    def set_obj(self, x_value, obj):
        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            self._objs[index] = obj
        else:
            self._dataFrames[index].set_obj(x_value[1:], obj)

    def increment_obj(self, x_value, increment):
        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            self._objs[index] += increment
        else:
            self._dataFrames[index].increment_obj(x_value[1:], increment)

    def get_obj_by_index(self, x_index):

        if self._ifOneDim:
            return self._objs[x_index[0]]
        else:
            return self._dataFrames[x_index[0]].get_obj_by_index(x_index[1:])

    def sample_indices(self, rng):
        """
        :param rng: random number generator
        :return: (list) indices of categories
                (in the example above, [1, 0] corresponds to age group 1 and sex group 0.
        """

        probs = []
        if self._ifOneDim:
            probs = self._objs
        else:
            for df in self._dataFrames:
                probs.append(df.get_sum())

        emprical_dist = RVGs.Empirical(probabilities=np.array(probs)/sum(probs))

        idx = emprical_dist.sample(rng=rng)

        if self._ifOneDim:
            return [idx]
        else:
            a = [idx]
            a.extend(self._dataFrames[idx].sample_indices(rng))
            return a

    def get_rows(self):
        rows = []
        if self._ifOneDim:
            for x in self._xs:
                rows.append([x, self.get_obj(x_value=[x])])
        else:
            for i, df in enumerate(self._dataFrames):
                next_rows = df.get_rows()
                for j, row in enumerate(next_rows):
                    a = [self._xs[i]]
                    a.extend(next_rows[j])
                    rows.append(a)

        return rows

    def get_objs(self):
        objs = []
        if self._ifOneDim:
            for x in self._xs:
                objs.append(self.get_obj(x_value=[x]))
        else:
            for df in self._dataFrames:
                objs.extend(df.get_objs())
        return objs

    def get_objs_gen(self):
        if self._ifOneDim:
            for x in self._xs:
                yield self.get_obj(x_value=[x])
        else:
            for df in self._dataFrames:
                for g in df.get_objs_gen():
                    yield g


class DataFrame(_DataFrame):
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        _DataFrame.__init__(self,
                            list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

        for row in rows:
            self.update_value(x_value=row[0:-1], v=row[-1])

    def get_value_by_index(self, x_index):

        return self.get_obj_by_index(x_index)

    def get_value(self, x_value):

        return self.get_obj(x_value)


class DataFrameWithExpDist(_DataFrame):
    """
    example:
        age,   sex,      mortality rate
        0,     0,        0.1,
        0,     1,        0.11,
        5,     0,        0.2,
        5,     1,        0.21,
        10,    0,        0.3
        10,    1,        0.31
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        _DataFrame.__init__(self,
                            list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

        for row in rows:
            self.update_value(x_value=row[0:-1],
                              v=RVGs.Exponential(scale=1/row[-1]))

    def get_dist(self, x_value):

        return self.get_obj(x_value)


class DataFrameWithEmpiricalDist(DataFrame):
    """
    example:
        age,   sex,      probability
        0,     0,        0.1,
        0,     1,        0.2,
        5,     0,        0.3,
        5,     1,        0.4,
    """
    def __init__(self, rows, list_x_min, list_x_max, list_x_delta):
        """
        :param rows: (list of list) the table above
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        DataFrame.__init__(self, rows=rows,
                           list_x_min=list_x_min,
                           list_x_max=list_x_max,
                           list_x_delta=list_x_delta)
        self.listXDelta = list_x_delta

        # make sure probabilities add to 1
        sum_probs = 0
        for row in rows:
            sum_probs += row[-1]
        if sum_probs < 0.99999 or sum_probs > 1.000001:
            raise ValueError('Sum of probabilities should add to 1.')

    def sample_values(self, rng):
        """
        :param rng: random number generator
        :return: (list) values of categories
                (in the example above, [1.4, 0] corresponds to age 1.4 and sex group 0.
        """

        values = []
        idx = self.sample_indices(rng=rng)
        for i, deltaX in enumerate(self.listXDelta):
            if deltaX == 'int':
                values.append(idx[i])
            else:
                unif_dist = RVGs.Uniform(loc=0, scale=deltaX)
                values.append(idx[i]*deltaX + unif_dist.sample(rng=rng))

        return values


class MortalityModel(_DataFrame):
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

    def __init__(self, rows, group_mins, group_max, group_delta, age_min, age_delta):
        """
        :param rows: (list of list) the table above
        :param group_mins: list of minimum value of x (in example above: [0, 0])
        :param group_max: list of maximum value of x (in example above: [10, 1])
        :param group_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        :param age_min:
        :param age_delta:
        """

        _DataFrame.__init__(self,
                            list_x_min=group_mins,
                            list_x_max=group_max,
                            list_x_delta=group_delta)

        self.ageMin = age_min

        for df_row in self.get_rows():
            rates = []
            for row in rows:
                if df_row[:-1] == row[1:-1]:
                    rates.append(row[-1])
            self.update_value(x_value=df_row[0:-1],
                              v=RVGs.NonHomogeneousExponential(rates=rates, delta_t=age_delta))

    def sample_time_to_death(self, group, age, rng):

        if age < self.ageMin:
            raise ValueError('Current age cannot be smaller than the minimum age.')

        return self.get_obj(group).sample(rng=rng, arg=age)


class Pyramid(_DataFrame):
    """
    example:
        age,   sex,      Prevalence
        0,     0,        10,
        0,     1,        20,
        5,     0,        30,
        5,     1,        40,
        10,    0,        50,
        10,    1,        60
    """

    def __init__(self, list_x_min, list_x_max, list_x_delta):
        """
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        _DataFrame.__init__(self,
                            list_x_min=list_x_min,
                            list_x_max=list_x_max,
                            list_x_delta=list_x_delta)

    def record_increment(self, x_values, increment):
        """
        updates the value of a group in the pyramid
        :param x_values: (list) group to be updated
                (e.g. [1.2, 0] corresponds to age 1.2 and sex 0 in the example above)
        :param increment: (integer) change (+ or -) in the value of the pyramid at the specified group
        """

        self.increment_obj(x_value=x_values, increment=increment)

    def record_value(self, x_values, value):
        """
        updates the value of a group in the pyramid
        :param x_values: (list) group to be updated
                (e.g. [1.2, 0] corresponds to age 1.2 and sex 0 in the example above)
        :param value: value of the pyramid at the specified group
        """

        self.set_obj(x_value=x_values, obj=value)

    def record_values_from_another_pyramid(self, another_pyramid):
        """ updates this pyramid using another pyramid """

        for row in another_pyramid.get_rows():
            self.record_value(x_values=row[:-1],
                              value=row[-1])

    def get_current_value(self, x_values):
        """
        :returns the value of the pyramid at the specified group
        :param x_values: (list) group
                (e.g. [1.2, 0] corresponds to age 1.2 and sex 0 in the example above)
        """

        return self.get_obj(x_value=x_values)

    def get_table_of_values(self):
        """ :returns the value of the pyramid in the table format
            In the example above, it returns:
            [
            [0,     0,        10],
            [0,     1,        20],
            [5,     0,        30],
            [5,     1,        40],
            [10,    0,        50],
            [10,    1,        60]
            ]

        """
        return self.get_rows()
