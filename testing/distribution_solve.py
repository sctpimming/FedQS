import math
import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, repeat
from tqdm import tqdm

from util.cspolicy import (
    client_sample_uni,
    client_sample_KL,
    client_sample_KLGap,
    client_sample_L2,
    client_sample_L1,
    client_sample_POC,
    client_sample_CBS,
)
from util.metric import accuracy, crossentropy
from util.misc import softmax, mod_softmax

np.random.seed(seed=12345)

grid = []

N = 10
m = int(0.3 * N)
d = 28 * 28
K = 10
B = 64
I = 30
T = 500

M = 500
C = 1 - (K * math.exp(-M))
mu = 1e-6
L = math.sqrt(2) + mu
gamma = (4 * L) / mu


def generate_grid():
    with open("data\grid_m3.json", "r") as f:
        grid = json.load(f)
    return grid["grid"]


def import_data():
    train_path = "data\FMNIST\FMNIST_alpha01_train.json"
    test_path = "data\FMNIST\FMNIST_alpha01_test.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


def main():
    print("hello")
    train_data, test_data = import_data()
    grid = generate_grid()
    print(len(grid))
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(
        [test_data["distribution"][uname] for uname in test_data["distribution"]]
    )
    test_dist = sum(test_dist) / N
    S = client_sample_L1(train_dist, test_dist, N, m, grid)
    print(S)


if __name__ == "__main__":
    main()
