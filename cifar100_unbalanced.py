import csv
from tqdm import tqdm
import math
from numpy.random import dirichlet
from numpy.random import choice
from util.misc import KL
import numpy as np
import json
import pickle
from matplotlib import pyplot as plt

d = 32 * 32 * 3
K = 100
N = 1000
alpha_train = 0.3
alpha_test = 0.1

X_train = []
Y_train = []
X_test = []
Y_test = []

data_train_path = "data/CIFAR100/cifar-100-python/train"
data_test_path = "data/CIFAR100/cifar-100-python/test"
with open(data_train_path, 'rb') as fo:
    data_train_dict = pickle.load(fo, encoding='bytes')
with open(data_test_path, 'rb') as fo:
    data_test_dict = pickle.load(fo, encoding='bytes')

def stack(data):
    X = []
    n, d = data.shape
    for i in range(n):
        feature = data[i]
        xsplit = [x.reshape((32, 32)) / 255.0 for x in np.split(feature, 3)]
        rgb = np.dstack((xsplit[0], xsplit[1], xsplit[2]))
        X.append(rgb)
    return X

X_train = stack(data_train_dict[b'data'])
Y_train = np.array(data_train_dict[b'fine_labels'])
X_test = stack(data_test_dict[b'data'])
Y_test = np.array(data_test_dict[b'fine_labels'])

print(data_train_dict[b"data"].shape)
print(len(X_train), len((Y_train)))
print(len(X_test), len((Y_test)))

combined = list(zip(X_train, Y_train))
sorted_train = sorted(combined, key=lambda x: x[1])
grouped_train = [[v[0] for v in sorted_train if v[1] == k] for k in range(K)]

print([len(grouped_train[k]) for k in range(K)])
combined = list(zip(X_test, Y_test))
sorted_test = sorted(combined, key=lambda x: x[1])
grouped_test = [[v[0] for v in sorted_test if v[1] == k] for k in range(K)]
# Split the data

for v in ["v1", "v2", "v3"]:
    train_path = f"data/Federated/CIFAR100_unbalanced_{v}_train.pck"
    test_path = f"data/Federated/CIFAR100_unbalanced_{v}_test.pck"
    print(f"Partition the {v} dataset with alpha = {alpha_test}")

    # Create data structure
    train_data = {"users": [], "user_data": {}, "distribution": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "distribution": {}}

    num_samples_train = np.clip(np.random.lognormal(4, 2, (N)).astype(int) + int((len(Y_train) / N)), 500, 1000)
    num_samples_test = int(len(Y_test) * 0.25)
    print("Number of sample")
    print(num_samples_train, num_samples_test)
    # Simulate the noniid-ness using dirichlet distribution
    distribution_train = dirichlet([alpha_train] * K, size=N)
    distribution_test = dirichlet([alpha_test] * K, size=1)

    test_label_n = choice(K, num_samples_test - K, p=distribution_test[0])
    major_class = np.bincount(test_label_n).argmax()
    print(f"The majority class is {major_class}")
    mask = np.ones(K, dtype=bool)
    mask[major_class] = 0
    class_list = np.arange(K)[mask]
    for n in range(N):
        uname = "f_{0:05d}".format(n)
        distribution_train[n][major_class] = 0
        train_label_n = choice(K, num_samples_train[n], p=distribution_train[n]/sum(distribution_train[n]))
        # print(train_label_n)
        train_n = [
            (grouped_train[k][choice(len(grouped_train[k]))], int(k)) for k in train_label_n
        ]
        
        if n <= 3:
            aux_num = 10
            for i in range(aux_num):
                train_n.append((grouped_train[major_class][choice(len(grouped_train[major_class]))], int(major_class)))
            num_samples_train += aux_num

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
        if n <= 5:
            print(label_dist)
        label_dist = label_dist / num_samples_train[n]

        train_data["distribution"][uname] = list(label_dist)

        class_list = np.array(
            [train_data["user_data"][uname]["y"].count(k) for k in range(10)]
        )



    # test data
    uname = "test"

    # test_label_n = choice(K, num_samples_test - K, p=distribution_test[0])
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
    train_dist = sum(train_dist) / N

    test_dist = np.array(test_data["distribution"]["test"])

    print(f"overall KL is {KL(test_dist, train_dist)}")

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)


