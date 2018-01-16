from scr import InOutFunctions as OutSupport

myList = [['Col1', 'Col2', 'Col3']]
for i in range(1, 10):
    myList.append([i, 2*i, 3*i])

OutSupport.write_csv('myCSV', myList)