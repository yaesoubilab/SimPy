import csv
import os
import math
import numpy as numpy
from collections import OrderedDict


def write_csv(rows, file_name='csvfile.csv', delimiter=',', directory='', delete_existing_files=False):
    """ write a list to a csv file
    :param rows: list of lists to be imported to the csv file
    :param file_name: the file name to be given to the csv file
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param directory: directory (relative to the current root) where the files should be located
            for example use 'Example' to create and save the csv file under the folder Example
    :param delete_existing_files: set to True to delete the existing trace files in the specified directory
    """

    # create a new file
    file_name = os.path.join(directory, file_name)

    # get directory
    directory_path = os.path.dirname(file_name)

    # delete existing files
    if delete_existing_files:
        delete_files(extension='.csv', path=os.getcwd() + '/' + directory)

    # create the directory if does not exist
    if directory_path != '':
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    with open(file_name, "w", newline='') as file:
        csv_file = csv.writer(file, delimiter=delimiter)

        for row in rows:
            csv_file.writerow(row)

        file.close()


def write_columns_to_csv(cols, file_name='csvfile.csv', delimiter=',', directory='', delete_existing_files=False):
    """ write a list of columns to a csv file
        :param cols: list of columns to be imported to the csv file
        :param file_name: the file name to be given to the csv file
        :param delimiter: to separate by comma, use ',' and by tab, use '\t'
        :param directory: directory (relative to the current root) where the files should be located
                for example use 'Example' to create and save the csv file under the folder Example
        :param delete_existing_files: set to True to delete the existing trace files in the specified directory
        """
    write_csv(
        rows=_cols_to_rows(cols=cols),
        file_name=file_name, delimiter=delimiter, directory=directory, delete_existing_files=delete_existing_files
    )


def read_csv_rows(file_name, if_ignore_first_row, delimiter=',', if_convert_float=False):
    """ reads the rows of a csv file
    :param file_name: the csv file name
    :param if_ignore_first_row: set true to ignore the first row
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param if_convert_float: set true to convert row values to numbers (otherwise, the values are stored as string)
    :returns a list containing the rows of the csv file
    """
    with open(file_name, "r") as file:
        csv_file = csv.reader(file, delimiter=delimiter)

        rows = _csv_file_to_rows(csv_file=csv_file,
                                 if_del_first_row=if_ignore_first_row)

        # convert column values to float if needed
        if if_convert_float:
            for i in range(0, len(rows)):
                rows[i] = _convert_to_float(rows[i])

        return rows


def read_csv_cols(file_name, n_cols, if_ignore_first_row, delimiter=',', if_convert_float=False):
    """ reads the columns of a csv file
    :param file_name: the csv file name
    :param n_cols: number of columns in the csv file
    :param if_ignore_first_row: set True to ignore the first row
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param if_convert_float: set True to convert column values to numbers
    :returns a list containing the columns of the csv file
    """
    with open(file_name, "r", encoding='utf-8', errors='ignore') as file:
        csv_file = csv.reader(file, delimiter=delimiter)

        cols = _rows_to_cols(rows=_csv_file_to_rows(csv_file=csv_file,
                                                    if_del_first_row=if_ignore_first_row),
                             n_cols=n_cols)

        # convert column values to float if needed
        if if_convert_float:
            for j in range(0, n_cols):
                cols[j] = _convert_to_float(cols[j])

        return cols


def read_csv_cols_to_dictionary(file_name, delimiter=',', if_convert_float=False):

    dict_of_columns = OrderedDict()  # dictionary of columns
    csv_file = open(file_name, "r", encoding='utf-8', errors='ignore')
    col_headers = next(csv.reader(csv_file, delimiter=delimiter))
    n_cols = len(col_headers)
    cols = read_csv_cols(
        file_name, 
        n_cols=n_cols,
        if_ignore_first_row=True,
        delimiter=delimiter,
        if_convert_float=if_convert_float)

    # add columns to the dictionary
    for j, col in enumerate(cols):
        if col_headers[j] in dict_of_columns:
            raise ValueError("Key '{}' already exists in the dictionary of parameters.".format(col_headers[j]))
        else:
            dict_of_columns[col_headers[j]] = col

    return dict_of_columns


def delete_files(extension='.txt', path='..'):
    """ delete every files with the specified extension inside the directory
    :param extension: (string) extension of the files to be removed
    :param path: (string) path (relative to the current root) where the files are located
    (the folder should already exist) use os.getcwd() to get the working directory
    """

    for f in os.listdir(path):
        if f.endswith(extension):
            os.remove(os.path.join(path, f))


def _csv_file_to_rows(csv_file, if_del_first_row):

    # read rows
    rows = []
    for i, row in enumerate(csv_file):
        rows.append(row)

    # delete the first row if needed
    if if_del_first_row:
        del rows[0]

    return rows


def _rows_to_cols(rows, n_cols):

    # initialize the list to store column values
    cols = []
    for j in range(0, n_cols):
        cols.append([])

    # read columns
    for row in rows:
        if len(row) != n_cols:
            raise ValueError('All rows should have the same length.')

        for j in range(0, n_cols):
            cols[j].append(row[j])

    return cols


def _cols_to_rows(cols):

    # find the size of the largest column
    size_of_largest_column = 0
    for col in cols:
        size = len(col)
        if size > size_of_largest_column:
            size_of_largest_column = size

    # initialize rows
    rows = []
    for i in range(0, size_of_largest_column):
        rows.append([])

    # populate rows
    for col in cols:
        for i, val in enumerate(col):
            rows[i].append(val)

    return rows


def _convert_to_float(list_of_objs):

    try:
        results = numpy.array(list_of_objs).astype(numpy.float)
    except:
        results = []
        for i in range(len(list_of_objs)):
            try:
                x = float(list_of_objs[i])
            except:
                if list_of_objs[i] in ('N/A', 'None', 'none', ''):
                    x = None
                else:
                    x = list_of_objs[i]
            results.append(x)

    return results
