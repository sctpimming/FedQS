import csv
from tqdm import tqdm
import math
from numpy.random import dirichlet
from numpy.random import choice
from util.misc import L1
import numpy as np
import pickle


def KL(P, Q):
    return sum(
        [P[k] * math.log(P[k] / Q[k]) for k in range(10) if (P[k] > 0 and Q[k] > 0)]
    )


train_path = "data/FMNIST/archive/fashion-mnist_train.csv"
test_path = "data/FMNIST/archive/fashion-mnist_test.csv"

d = 28 * 28
K = 10
N = 1000
alpha_train = 0.3
alpha_test = 0.3

X_train = []
Y_train = []
X_test = []
Y_test = []

with open(train_path, "r") as data:
    reader = csv.reader(data)
    next(reader)
    for line in reader:
        row = np.array([int(v) for v in line])
        feature = np.array(row[:-1]) / 255.0
        X_train.append(feature.reshape((28, 28, 1)))
        Y_train.append(row[0])

with open(test_path, "r") as data:
    reader = csv.reader(data)
    next(reader)
    for line in reader:
        row = np.array([int(v) for v in line])
        feature = np.array(row[:-1]) / 255.0
        X_test.append(feature.reshape((28, 28, 1)))
        Y_test.append(row[0])

# Combine, sort data by label, grouped data by label

combined = list(zip(X_train, Y_train))
sorted_train = sorted(combined, key=lambda x: x[1])
grouped_train = [[v[0] for v in sorted_train if v[1] == k] for k in range(K)]

combined = list(zip(X_test, Y_test))
sorted_test = sorted(combined, key=lambda x: x[1])
grouped_test = [[v[0] for v in sorted_test if v[1] == k] for k in range(K)]

# Split the data

for v in ["v5"]:
    train_path = f"data/Federated/FMNIST_alpha05_{v}_train.pck"
    test_path = f"data/Federated/FMNIST_alpha05_{v}_test.pck"
    print(f"Partition the {v} dataset with alpha = {alpha_train}")

    # Create data structure
    train_data = {"users": [], "user_data": {}, "distribution": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "distribution": {}}

    # TODO: Implement varied number of samples
    # num_samples_train = int(len(Y_train) / N)
    num_samples_train = np.random.lognormal(4, 2, (N)).astype(int) + int(len(Y_train) / N)
    num_samples_test = 1000
    # print(num_samples_train, num_samples_test)
    # Simulate the noniid-ness using dirichlet distribution
    distribution_train = dirichlet([alpha_train] * K, size=N)
    distribution_test = dirichlet([alpha_test] * K, size=1)

    for n in range(N):
        uname = "f_{0:05d}".format(n)
        train_label_n = choice(K, num_samples_train[n], p=distribution_train[n])
        train_n = [
            (grouped_train[k][choice(len(grouped_train[k]))], int(k)) for k in train_label_n
        ]

        train_data["users"].append(uname)
        train_data["user_data"][uname] = {
            "x": [v[0] for v in train_n],
            "y": [v[1] for v in train_n],
        }
        train_data["num_samples"].append(int(num_samples_train[n]))

        label_dist = np.zeros(K)
        for v in train_n:
            label = v[1]
            label_dist[label] += 1
        label_dist = label_dist / num_samples_train[n]

        train_data["distribution"][uname] = list(label_dist)

        class_list = np.array(
            [train_data["user_data"][uname]["y"].count(k) for k in range(10)]
        )
        # print(class_list)

    # test data
    uname = "test"
    test_label_n = choice(K, num_samples_test - K, p=distribution_test[0])
    test_n = [(grouped_test[k][choice(len(grouped_test[k]))], int(k)) for k in test_label_n]

    # guarantee non-zero PMF
    for k in range(K):
        test_n.append((grouped_test[k][choice(len(grouped_test[k]))], int(k)))

    test_data["users"].append(uname)
    test_data["user_data"][uname] = {
        "x": [v[0] for v in test_n],
        "y": [v[1] for v in test_n],
    }
    label_dist = np.zeros(K)
    for v in test_n:
        label = v[1]
        label_dist[label] += 1
    print(label_dist)
    label_dist = label_dist / num_samples_test
    print(label_dist)
    test_data["distribution"][uname] = list(label_dist)


    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(test_data["distribution"]["test"])

    train_dist = sum(train_dist) / N

    print(f"overall KL is {KL(test_dist, train_dist)}")

    train_pck = open(train_path, "wb")
    test_pck = open(test_path, "wb")

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)
