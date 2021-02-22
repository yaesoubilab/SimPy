
def index_of_an_action_combo(action_combo):

    s = 0
    n = len(action_combo)
    for i in range(n):
        s += action_combo[i] * pow(2, n-i-1)
    return int(s)


def action_combo_of_an_index(index):

    string = bin(index).replace("0b", "")
    return list(map(int, string))
