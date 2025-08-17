import matplotlib.pyplot as plt
from typing import Sequence,Any
from collections import defaultdict

from helper import Block
from even import count_pairs


def pretty_plot_races(races:list[Block]) -> None:
    B = len(races)
    headers = [f"Race {b}" for b in range(1,B+1)]
    # to work with data, we need to transpose it
    data = [sorted(block) for block in races]
    transposed = []
    for col in range(len(data[0])):
        new_row = []
        for row in range(B):
            if len(data[row]) <= col:
                new_row.append('')
            else:
                new_row.append(str(data[row][col]))
        transposed.append(new_row)
    data = transposed
    # print(data)
    # print(headers)

    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Hide the axes to display only the table
    ax.axis('off')

    # Create the table
    headerColour = (255 / 255, 213 / 255, 97 / 255)
    table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center', colColours=[headerColour] * B)

    # Adjust table properties for better appearance (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Display the table
    plt.show()


def print_player_matrix(roster:Sequence[Any], races:list[Block]) -> None:
    import perms
    matrix = defaultdict(lambda: defaultdict(int))
    max_cnt = 0
    all_pairs = list(perms.combs(roster, 2, True))
    PC = count_pairs(races, all_pairs)
    for pair,cnt in PC.items():
        if cnt > max_cnt:
            max_cnt = cnt
        matrix[pair[0]][pair[1]] = cnt
        matrix[pair[1]][pair[0]] = cnt
    V = len(roster)
    Vlen = len(str(V))
    CntLen = len(str(max_cnt))
    NameLen = max([len(str(name)) for name in roster])
    sizing = max([Vlen, CntLen, NameLen])
    print(' ' * sizing + ' ' + ' '.join(f"{name:^{sizing}}" for name in roster))
    # print('-+-'.join('-'*sizing for _ in range(V)))
    for name1 in roster:
        print(f'{name1:>{sizing}} ', end='')
        values = []
        for name2 in roster:
            v = matrix[name1][name2]
            if name1 == name2:
                if v > 0:
                    v = f'*{v}'
                else:
                    v = '-'
            values.append(v)
        # values = [matrix[name1][name2] if name1 != name2 else (f'*{matrix[name1][name2]}' if matrix[name1][name2] > 0 '-') for name2 in roster]
        print(' '.join(f"{v:^{sizing}}" for v in values))
    print()
