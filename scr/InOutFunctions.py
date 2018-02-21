import csv
import numpy as numpy


def write_csv(file_name, rows, delimiter='\t'):
    """ write a list to a csv file
    :param file_name: the file name to be given to the csv file
    :param rows: list of lists to be imported to the csv file
    :param delimiter: to separate by comma, use ',' and by tab, use '\t'
    """
    with open(file_name, "w", newline='') as file:
        csv_file = csv.writer(file, delimiter=delimiter)  # use '\t' for tab

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
        csv_file = csv.reader(file, delimiter=delimiter)  # use '\t' for tab

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
                rows[i] = numpy.array(rows[i]).astype(numpy.float)

        return rows


def read_csv_cols(file_name, n_cols, if_ignore_first_row, if_convert_float):
    """ reads the columns of a csv file
    :param file_name: the csv file name
    :param n_cols: number of columns in the csv file
    :param if_ignore_first_row: set True to ignore the first row
    :param if_convert_float: set True to convert column values to numbers
    :returns a list containing the columns of the csv file
    """
    with open(file_name, "r") as file:
        csv_file = csv.reader(file, delimiter='\t')  # use '\t' for tab

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
                cols[j] = numpy.array(cols[j]).astype(numpy.float)

        return cols
