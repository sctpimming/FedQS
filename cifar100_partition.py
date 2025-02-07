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
N = 5000
alpha_train = 0.1
alpha_test = 0.05
epsilon = 1/K

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

for v in ["viz"]:
    train_path = f"data/Federated/CIFAR100_alpha005_{v}_train.pck"
    test_path = f"data/Federated/CIFAR100_alpha005_{v}_test.pck"
    print(f"Partition the {v} dataset with alpha = {alpha_test}")

    # Create data structure
    train_data = {"users": [], "user_data": {}, "distribution": {}, "num_samples": []}
    test_data = {"users": [], "user_data": {}, "distribution": {}}

    num_samples_train = np.random.lognormal(4, 2, (N)).astype(int)
    num_samples_train = np.clip(num_samples_train, 2, 500)

    num_samples_test = 500
    print("Number of sample")
    print(num_samples_train, num_samples_test)
    # Simulate the noniid-ness using dirichlet distribution
    alpha_vec = np.linspace(0.01, 0.3, K)
    # alpha_vec = np.multiply(np.ones(K), [0.95**k for k in range(K)])
    alpha_vec = [alpha_train]*K
    np.random.shuffle(alpha_vec)
    # print(alpha_vec)
    distribution_train = dirichlet(alpha_vec, size=N)
    distribution_test = dirichlet([alpha_test] * K, size=1)
    name_list = ["f_{0:05d}".format(n) for n in range(N)]
    for n in range(N):
        uname = "f_{0:05d}".format(n)
        train_label_n = choice(K, num_samples_train[n], p=distribution_train[n])
        # print(train_label_n)
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
        # for k in range(K):
        #     if label_dist[k] == 0:
        #         label_dist[k] = epsilon

        train_data["distribution"][uname] = list(label_dist/sum(label_dist))
        
        class_list = np.array(
            [train_data["user_data"][uname]["y"].count(k) for k in range(10)]
        )



    # test data
    uname = "test"
    train_dist = np.array([train_data["distribution"][uname] for uname in name_list])
    mixture_weight = np.random.power(0.001, N)
    mixture_weight = mixture_weight/sum(mixture_weight)
    test_dist = np.matmul(mixture_weight, train_dist)
    train_dist = sum(train_dist)/N

    test_dist = distribution_test[0]
    show_hist(train_dist, test_dist)
    print(f"overall KL is {KL(test_dist, train_dist)}")
    # print(sum(test_dist), sum(train_dist))
    # test_dist = train_dist[0]
    distribution_test = [test_dist]
    test_label_n = choice(K, num_samples_test, p=distribution_test[0])
    freq = np.zeros(K)
    for k in test_label_n:
        freq[k] += 1
    # print(freq)
    sample_idx_list = [
        choice(len(grouped_test[k]), min(int(freq[k]), 100), replace=False) 
        for k in range(K)
    ]
    # for k in range(K):
    #     print(sample_idx_list[k])
    # test_n.append((grouped_test[k][choice(len(grouped_test[k]),)], int(k)))
    test_n = []
    for k in range(K):
        for idx in sample_idx_list[k]:
            test_n.append((grouped_test[k][idx], int(k)))
    # test_n = [(grouped_test[k][sample_idx_list[k]], int(k)) for k in range(K)]

    # guarantee non-zero PMF
    # for k in range(K):
    #     test_n.append((grouped_test[k][choice(len(grouped_test[k]))], int(k)))

    test_data["users"].append(uname)
    test_data["user_data"][uname] = {
        "x": [v[0] for v in test_n],
        "y": [v[1] for v in test_n],
    }
    label_dist = np.zeros(K)
    for v in test_n:
        label = v[1]
        label_dist[label] += 1
    # print(label_dist)
    label_dist = label_dist / num_samples_test
    # print(label_dist)
    # for k in range(K):
    #     if label_dist[k] == 0:
    #         label_dist[k] = epsilon
    test_data["distribution"][uname] = list(label_dist/sum(label_dist))
    


    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    train_dist = sum(train_dist) / N

    test_dist = np.array(test_data["distribution"]["test"])
    show_hist(train_dist, test_dist)


    print(f"overall KL is {KL(test_dist, train_dist)}")

    with open(train_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(test_path, "wb") as f:
        pickle.dump(test_data, f)


