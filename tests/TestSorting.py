

def takeSecond(elem):
    return elem[1]


list = [
    [1, 2],
    [2, 4],
    [3, 3],
]

print(list)
list.sort(key=takeSecond)

print(list)
subset = [row for row in list if row[0]==2]

print(subset)
subset[0][1] = 23

print(subset)
print(list)