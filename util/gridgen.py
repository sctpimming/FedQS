from itertools import product
import numpy as np
import json
import multiprocessing
from tqdm import tqdm

N = 10
sz = 5
m = 3

cartesian = product(np.linspace(0, 1, sz), repeat=N)
grid = [
    point
    for point in tqdm(cartesian)
    if sum(point) == 3 and np.count_nonzero(point) >= int(0.3 * N)
]

print(len(grid))
output_grid = {"grid": grid}
with open("data\grid_m3.json", "w") as f:
    json.dump(output_grid, f)
