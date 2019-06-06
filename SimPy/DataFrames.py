import SimPy.RandomVariantGenerators as RVGs
import numpy as np
import math


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
                but it could also be a list of any objects.
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
                return math.floor((x_value-self.xMin)/self.xDelta)

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
        age,    exponential dist. with mortality rates
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

        # create the list of exponential distributions
        y_objects = []
        for v in y_values:
            if v <= 0:
                raise ValueError('All y_values (rates of exponential distributions) should be greater than 0.')
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


class DataFrameOfObjects:
    """
    example:
        age,   sex,      objects
        0,     0,        obj_0,
        0,     1,        obj_1,
        5,     0,        obj_2,
        5,     1,        obj_3,
        10,    0,        obj_4,
        10,    1,        obj_5
    """

    def __init__(self, list_x_min, list_x_max, list_x_delta, name=''):
        """
        :param list_x_min: list of minimum values of x (in example above: [0, 0] for age and sex)
        :param list_x_max: list of maximum values of x (in example above: [10, 1] for age and sex)
        :param list_x_delta: list of interval between break points of x
                             if set to 'int', x is treated as categorical variable
                             (in example above: [5, 'int'] for age and sex)
        :param name: name of this data frame
        """

        # if x is 1 dimensional, convert min, max, delta into lists
        if not isinstance(list_x_min, list):
            list_x_min = [list_x_min]
        if not isinstance(list_x_max, list):
            list_x_max = [list_x_max]
        if not isinstance(list_x_delta, list):
            list_x_delta = [list_x_delta]

        self._xMin = list_x_min[0]
        self._xMax = list_x_max[0]
        self._xDelta = list_x_delta[0]
        self.name = name

        # This class creates (recursively) nested data frames which allow for
        # quick access to the objects given x values.
        # In the example above, it uses the first dimension of x to create a list of data frames.
        # [df_1, df_2, df_3] which correspond to age group [0, 5), [5, 10), and [10+].
        #       df_1 will be data frame over sex with [obj_0, obj_1] as objects.
        #       df_2 will be data frame over sex with [obj_2, obj_3] as objects.
        #       df_2 will be data frame over sex with [obj_4, obj_5] as objects.

        # if this is a 1-dimensional data frame (example above is a 2-dimensional data frame)
        self._ifOneDim = False  # in the example above, False for age category and True for sex category

        # the variables below stores information on the first dimension of x
        # in the example above, it is age.
        self._xs = []               # [0, 5, 10]
        self._ifContinuous = True   # age is a continuous variable

        # find if it is a 1-dimensional data frame
        if len(list_x_min) == 1:
            self._ifOneDim = True
        # find if the first dimensions is a categorical variable
        if list_x_delta[0] == 'int':
            self._ifContinuous = False

        # find x breakpoints of the first dimension
        x = list_x_min[0]
        while x <= list_x_max[0]:
            self._xs.append(x)
            if self._ifContinuous:
                x += list_x_delta[0]
            else:
                x += 1

        # to store y-values:
        # if it is a 1-dimensional data frame, this will contain numbers
        # if it is a multi-dimensional data frame, this will contain data frames that
        # correspond to second, third, etc. dimensions
        self._objs = []
        for x in self._xs:
            if self._ifOneDim:
                self._objs.append(0)
            else:
                self._objs.append(
                    DataFrameOfObjects(list_x_min=list_x_min[1:],
                                       list_x_max=list_x_max[1:],
                                       list_x_delta=list_x_delta[1:]
                                       )
                )

    def __get_index(self, x_value):
        """
        :param x_value: (list) of x values (e.g. value of age and sex in the example above)
        :return: the index of the smallest x break point with x_value[0] greater than or equal.
        """

        # convert to list if x_value is not a list
        if not isinstance(x_value, list):
            x_value = [x_value]

        if not self._ifContinuous:
            if not (isinstance(x_value[0], np.integer) or isinstance(x_value[0], int)):
                raise ValueError('x_value should be an integer for categorical variables.')
            return int(x_value[0] - self._xMin)
        else:
            if x_value[0] > self._xMax:
                return len(self._objs) - 1
            else:
                return int(math.floor((x_value[0] - self._xMin) / self._xDelta))

    def get_sum(self):
        """
        :return: the sum of all objects stored in this data frame (assumes that objects are of type int or float).
        """
        if self._ifOneDim:
            return sum(self._objs)
        else:
            sum_df = 0
            for df in self._objs:
                sum_df += df.get_sum()
            return sum_df

    def get_percents(self):
        """
        :return: (list of lists of lists of ...)
                 Assuming that objects are of type int or float, it returns
                 the object divided by the sum of all objects.
                 in the example above, it returns:
                 [
                    [0, 0, 0.11904761904761904],
                    [0, 1, 0.23809523809523808],
                    [5, 0, 0.13095238095238096],
                    [5, 1, 0.14285714285714285],
                    [10, 0, 0.17857142857142858],
                    [10, 1, 0.19047619047619047]
                ]
        """
        total = self.get_sum()
        rows = []
        for row in self.get_rows():
            a = row[:-1]
            a.extend([row[-1]/total])
            rows.append(a)

        return rows

    def get_index(self, x_value):
        """
        :param x_value: (list) of x values (e.g. value of age and sex in the example above)
        :return: (list) indices that correspond to x values
            in the example above: it returns [0, 1] for 0<= x_value[0]<5 and x_value[1] = 1
        """

        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            return [index]
        else:
            return self._objs[index].get_index(x_value=x_value[1:])

    def get_obj(self, x_value):
        """
        :param x_value: (list) of x values (e.g. value of age and sex in the example above)
        :return: the object that correspond to x values
            in the example above: it returns obj_1 for 0<= x_value[0]<5 and x_value[1] = 1
        """

        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            return self._objs[index]
        else:
            return self._objs[index].get_obj(x_value[1:])

    def set_obj(self, x_value, obj):
        """
        set the object that correspond to x_value to obj
        :param x_value: (list) of x values (e.g. values of age and sex in the example above)
        :param obj: the new object
        """
        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            self._objs[index] = obj
        else:
            self._objs[index].set_obj(x_value=x_value[1:], obj=obj)

    def increment_obj(self, x_value, increment):
        """
        increments the value that correspond to x_value by increment
        :param x_value: (list) of x values (e.g. values of age and sex in the example above)
        :param increment: increment value
        """
        index = self.__get_index(x_value=x_value)
        if self._ifOneDim:
            self._objs[index] += increment
        else:
            self._objs[index].increment_obj(x_value[1:], increment)

    def get_obj_by_index(self, x_index):
        """
        :param x_index: (list) of indices of x values (e.g. index of age and sex in the example above)
        :return: the object that correspond to the indices of x values
            in the example above: it returns obj_1 for x_value [0, 1]
        """
        if self._ifOneDim:
            return self._objs[x_index[0]]
        else:
            return self._objs[x_index[0]].get_obj_by_index(x_index[1:])

    def sample_indices(self, rng):
        """
        :param rng: random number generator
        :return: (list) indices of sampled categories
                        (in the example above, [1, 0] corresponds to age group 1 and sex group 0.
        """

        probs = []
        if self._ifOneDim:
            probs = self._objs
        else:
            for df in self._objs:
                probs.append(df.get_sum())

        empirical_dist = RVGs.Empirical(probabilities=np.array(probs)/sum(probs))

        idx = empirical_dist.sample(rng=rng)

        if self._ifOneDim:
            return [idx]
        else:
            a = [idx]
            a.extend(self._objs[idx].sample_indices(rng))
            return a

    def get_rows(self):
        """
        :return: the data frame in rows. In the example above:
                [
                    [0,     0,        obj_0],
                    [0,     1,        obj_1],
                    [5,     0,        obj_2],
                    [5,     1,        obj_3],
                    [10,    0,        obj_4],
                    [10,    1,        obj_5]
                ]
        """
        rows = []
        if self._ifOneDim:
            for x in self._xs:
                rows.append([x, self.get_obj(x_value=[x])])
        else:
            for i, df in enumerate(self._objs):
                next_rows = df.get_rows()
                for j, row in enumerate(next_rows):
                    a = [self._xs[i]]
                    a.extend(next_rows[j])
                    rows.append(a)

        return rows

    def get_objs(self):
        """
        :return: (list) of objects stored in this data frame
        """
        objs = []
        if self._ifOneDim:
            for x in self._xs:
                objs.append(self.get_obj(x_value=[x]))
        else:
            for df in self._objs:
                objs.extend(df.get_objs())
        return objs

    def get_objs_gen(self):
        """
        :return: a generator function to go over all objects store in this data frame
        """
        if self._ifOneDim:
            for x in self._xs:
                yield self.get_obj(x_value=[x])
        else:
            for df in self._objs:
                for g in df.get_objs_gen():
                    yield g


class DataFrame(DataFrameOfObjects):
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

        DataFrameOfObjects.__init__(self,
                                    list_x_min=list_x_min,
                                    list_x_max=list_x_max,
                                    list_x_delta=list_x_delta)

        for row in rows:
            self.set_obj(x_value=row[0:-1], obj=row[-1])

    def get_value_by_index(self, x_index):

        return self.get_obj_by_index(x_index)

    def get_value(self, x_value):

        return self.get_obj(x_value)


class DataFrameWithExpDist(DataFrameOfObjects):
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

        DataFrameOfObjects.__init__(self,
                                    list_x_min=list_x_min,
                                    list_x_max=list_x_max,
                                    list_x_delta=list_x_delta)

        for row in rows:
            self.set_obj(x_value=row[0:-1],
                         obj=RVGs.Exponential(scale=1/row[-1]))

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


class Pyramid(DataFrameOfObjects):
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

    def __init__(self, list_x_min, list_x_max, list_x_delta, name=''):
        """
        :param list_x_min: list of minimum value of x (in example above: [0, 0])
        :param list_x_max: list of maximum value of x (in example above: [10, 1])
        :param list_x_delta: list of interval between break points of x
                    if set to 'int', x is treated as categorical variable
                    (in example above: [5, 'int'])
        """

        DataFrameOfObjects.__init__(self,
                                    list_x_min=list_x_min,
                                    list_x_max=list_x_max,
                                    list_x_delta=list_x_delta,
                                    name=name)

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

    def get_values(self):
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

    def get_percentages(self):
        """ :returns the percentage of the population in each group
            In the example above, it returns:
            [
            [0,     0,        10/210],
            [0,     1,        20/210],
            [5,     0,        30/210],
            [5,     1,        40/210],
            [10,    0,        50/210],
            [10,    1,        60/210]
            ]

        """
        return self.get_percents()
