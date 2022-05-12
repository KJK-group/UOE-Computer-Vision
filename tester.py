import torch
device = 'cpu'


def string_to_list(string):
    li = list(string.split(" "))
    return li


colors = ['[0 0 0]', '[ 19  77 188]', '[24  1 77]', '[ 6 35 18]', '[ 12  16 162]',
          '[243   5  42]', '[116 132   3]', '[ 1 58 15]', '[245  94  55]']

color_map = []
for color in colors:
    print(color)
    RGB_list = string_to_list(color)
    print(RGB_list)
    color_map.append(torch.tensor(RGB_list, device=device))

print(color_map)
