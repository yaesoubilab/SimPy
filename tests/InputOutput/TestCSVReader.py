from scr import InOutFunctions as InOutSupport

# test reading by rows
rows = InOutSupport.read_csv_rows('myCSV', if_del_first_row=True, if_convert_float=True)
print('Testing reading by rows:')
for row in rows:
    print(sum(row))

# test reading by columns
cols = InOutSupport.read_csv_cols('myCSV', n_cols=3, if_ignore_first_row=True, if_convert_float=True)
print('Testing reading by columns:')
for j in range(0, 3):
    print(sum(cols[j]))