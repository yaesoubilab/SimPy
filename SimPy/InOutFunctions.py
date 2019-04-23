import csv
import os

import numpy as numpy


def write_csv(rows, file_name='csvfile.csv', delimiter='\t', directory='', delete_existing_files=False):
    """ write a list to a csv file
    :param rows: list of lists to be imported to the csv file
    :param file_name: the file name to be given to the csv file
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param directory: directory (relative to the current root) where the files should be located
            for example use 'Example' to create and save the csv file under the folder Example
    :param delete_existing_files: set to True to delete the existing files in the directory
    """

    if directory != '':
        # create the directory if does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)

    if delete_existing_files:
        delete_files(extension='.txt', path=os.getcwd() + '/' + directory)

    # create a new file
    file_name = os.path.join(directory, file_name)

    with open(file_name, "w", newline='') as file:
        csv_file = csv.writer(file, delimiter=delimiter)

        for row in rows:
            csv_file.writerow(row)

        file.close()


def read_csv_rows(file_name, if_del_first_row, delimiter='\t', if_convert_float=False):
    """ reads the rows of a csv file
    :param file_name: the csv file name
    :param if_del_first_row: set true to delete the first row
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param if_convert_float: set true to convert row values to numbers (otherwise, the values are stored as string)
    :returns a list containing the rows of the csv file
    """
    with open(file_name, "r") as file:
        csv_file = csv.reader(file, delimiter=delimiter)

        # read rows
        rows = []
        for row in csv_file:
            rows.append(row)

        # delete the first row if needed
        if if_del_first_row:
            del rows[0]

        # convert column values to float if needed
        if if_convert_float:
            for i in range(0, len(rows)):
                try:
                    rows[i] = numpy.array(rows[i]).astype(numpy.float)
                except:
                    pass

        return rows


def read_csv_cols(file_name, n_cols, if_ignore_first_row, delimiter='\t', if_convert_float=False):
    """ reads the columns of a csv file
    :param file_name: the csv file name
    :param n_cols: number of columns in the csv file
    :param if_ignore_first_row: set True to ignore the first row
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    :param if_convert_float: set True to convert column values to numbers
    :returns a list containing the columns of the csv file
    """
    with open(file_name, "r") as file:
        csv_file = csv.reader(file, delimiter=delimiter)

        # initialize the list to store column values
        cols = []
        for j in range(0, n_cols):
            cols.append([])

        # read columns
        for row in csv_file:
            for j in range(0, n_cols):
                cols[j].append(row[j])

        # delete the first row if needed
        if if_ignore_first_row:
            for j in range(0, n_cols):
                del cols[j][0]

        # convert column values to float if needed
        if if_convert_float:
            for j in range(0, n_cols):
                try:
                    cols[j] = numpy.array(cols[j]).astype(numpy.float)
                except:
                    pass

        return cols


def read_csv_cols_to_dictionary(file_name, delimiter='\t', if_convert_float=False):

    dict_of_columns = {} # dictionary of columns   
    csv_file = open(file_name, "r")
    col_headers = next(csv.reader(csv_file, delimiter=delimiter))
    n_cols = len(col_headers)
    cols = read_csv_cols(
        file_name, 
        n_cols=n_cols,
        if_ignore_first_row=True,
        delimiter=delimiter,
        if_convert_float=if_convert_float)
    
    for j, col in enumerate(cols):
        dict_of_columns[col_headers[j]] = col

    return dict_of_columns


def delete_files(extension='.txt', path='..'):
    """ delete every files with the specified extension inside the directory
    :param extension: (string) extension of the files to be removed
    :param path: (string) path (relative to the current root) where the files are located
    (the folder should already exist)
    """

    for f in os.listdir(path):
        if f.endswith(extension):
            os.remove(os.path.join(path, f))
