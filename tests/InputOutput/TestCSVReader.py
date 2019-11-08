from SimPy import InOutFunctions as io

# ---------------
# first run the TestCSVWritter.py to produce the csv file
# ---------------

# test reading by rows
rows = io.read_csv_rows('CSVFolder/myCSV.csv',
                        if_ignore_first_row=True,
                        if_convert_float=True)
print('Testing reading by rows:')
for row in rows:
    print(sum(row[1:]))

# test reading by columns
cols = io.read_csv_cols('CSVFolder/myCSV.csv',
                        n_cols=4,
                        if_ignore_first_row=True,
                        if_convert_float=True)
print('Testing reading by columns:')
for j in range(1, 4):
    print(sum(cols[j]))

# rest reading by columns into a dictionary
dict_cols = io.read_csv_cols_to_dictionary(
    'CSVFolder/myCSV.csv',
    if_convert_float=True)

print('Testing reading by columns into a dictionary:')
print(dict_cols)