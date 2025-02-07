import csv
from tqdm import tqdm
import math
from numpy.random import dirichlet
from numpy.random import choice
import matplotlib.pyplot as plt
from util.misc import KL
import numpy as np
import json
import pickle

train_path = "data/CIFAR10/CIFAR10_train.csv"
test_path = "data/CIFAR10/CIFAR10_test.csv"

d = 32 * 32 * 3
K = 10
N = 200
alpha_train = 0.1
alpha_test = 0.3

def show_hist(train_dist, test_dist):
    K = len(test_dist)
    class_name = [f"{n}" for n in range(K)]
    bar_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:grey",
        "tab:olive",
        "tab:cyan",
    ]
    ylim = max(test_dist)
    plt.subplot(1, 2, 1)
    plt.bar(class_name, train_dist, color=bar_colors, width=0.9)
    plt.ylim([0, ylim])

    plt.subplot(1, 2, 2)
    plt.bar(class_name, test_dist, color=bar_colors, width=0.9)
    plt.ylim([0, ylim])

    plt.show()
    
def get_partition(data, label_list, uname):
    freq = np.zeros(K, dtype=int)
    for k in label_list:
        freq[k] += 1
    print(freq)
    if uname == "test":
        max_img_perclass = 1000
    else: 
        max_img_perclass = 4000
    sample_idx_list = [
        choice(len(data[k]), min(freq[k], len(data[k])), replace=False) 
        for k in range(K)
    ]
    x = [
        data[k][idx][0]
        for k in range(K)
        for idx in sample_idx_list[k]
    ]
    y = [
        int(data[k][idx][1]) 
        for k in range(K)
        for idx in sample_idx_list[k]
    ]
    label_dist = np.zeros(K)
    for v in y:
        label = v
        label_dist[label] += 1
    label_dist = label_dist / len(y)
    print(label_dist)
    return x, y, label_dist, len(y)


X = []
Y = []



with open(train_path, "r") as data:
    reader = csv.reader(data)
    next(reader)
    for line in reader:
        row = np.array([int(v) for v in line])

        feature = np.array(row[:-1])
        xsplit = [x.reshape((32, 32)) / 255.0 for x in np.split(feature, 3)]
        rgb = np.dstack((xsplit[0], xsplit[1], xsplit[2]))

        X.append(rgb)
        Y.append(row[-1])


nsamples = len(X)
train_ratio = 0.8
train_len = int(train_ratio * nsamples)

X_train = X[:train_len][:]
Y_train = Y[:train_len][:]
X_test = X[train_len:][:]
Y_test = Y[train_len:][:]

combined = list(zip(X_train, Y_train))
sorted_train = sorted(combined, key=lambda x: x[1])
grouped_train = [[v for v in sorted_train if v[1] == k] for k in range(K)]

print(len(grouped_train[0]))
combined = list(zip(X_test, Y_test))
sorted_test = sorted(combined, key=lambda x: x[1])
grouped_test = [[v for v in sorted_test if v[1] == k] for k in range(K)]

# Split the data

for v in ["control"]:
    train_path = f"data/Federated/CIFAR_alpha01_{v}_train.pck"
    test_path = f"data/Federated/CIFAR_alpha01_{v}_test.pck"
    print(f"Partition the {v} dataset with alpha = {alpha_test}")

    # Create data structure
    train_data = {"users": [], "user_data": {}, "distribution": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "distribution": {}}

    num_samples_train = np.random.lognormal(4, 2, (N)).astype(int) 
    num_samples_train = np.clip(num_samples_train, 50, 500)
    num_samples_test = int(nsamples * 0.25)

    # Simulate the noniid-ness using dirichlet distribution
    
    distribution_train = dirichlet([alpha_train]* K, size=N)
    distribution_test = dirichlet([alpha_test] * K, size=1)
    if v == "control":
        # Target distribution in uniform
        distribution_test = [[1/K]*K]

    for n in range(N):
        uname = "f_{0:05d}".format(n)
        train_label_n = choice(K, num_samples_train[n], p=distribution_train[n])
        x_n, y_n, label_dist, num_samples = get_partition(grouped_train, train_label_n, uname)      

        train_data["users"].append(uname)
        train_data["user_data"][uname] = {
            "x": x_n,
            "y": y_n,
        }

        train_data["distribution"][uname] = list(label_dist)
        train_data["num_samples"].append(num_samples)

    # test data
    uname = "test"
    test_label_n = choice(K, num_samples_test, p=distribution_test[0])
    x_n, y_n, label_dist, num_samples = get_partition(grouped_test, test_label_n, uname) 
    test_data["users"].append(uname)
    test_data["user_data"][uname] = {
        "x": x_n,
        "y": y_n,
    }
    test_data["distribution"][uname] = list(label_dist)


    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    train_dist = sum(train_dist) / N

    test_dist = np.array(test_data["distribution"]["test"])

    print(f"overall KL is {KL(test_dist, train_dist)}")
    show_hist(train_dist, test_dist)
    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
