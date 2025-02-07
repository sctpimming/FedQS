import math
import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, repeat
from tqdm import tqdm

from util.cspolicy import (
    client_sample_uni,
    client_sample_QS,
    client_sample_L1,
    client_sample_POC,
    client_sample_CBS,
)
from util.metric import accuracy, crossentropy
from util.misc import softmax, mod_softmax, L1

np.random.seed(seed=12345)

grid = []

N = 100
r = 30
m = 3
d = 28 * 28
K = 10
B = 64
I = 30
T = 200

M = 500
C = 1 - (K * math.exp(-M))
mu = 1e-6
L = math.sqrt(2) + mu
gamma = (4 * L) / mu
eta = 0.01
decay = 0.9992
test_size = 600


def import_data():
    train_path = "data/FMNIST/FMNIST_N100alpha01_train.json"
    test_path = "data/FMNIST/FMNIST_N100alpha01_test.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


def generate_grid():
    with open("data\grid.json", "r") as f:
        grid = json.load(f)
    return grid["grid"]


def init():
    global train_data
    global test_data
    train_data, test_data = import_data()


def gradient_eval(w, data, C, M):  # TODO: Check gradient correctness
    sz = len(data["y"])
    batch_idx = np.random.choice(sz, B, replace=False)
    gradient = np.zeros((K, d))
    for i in range(K):
        for n in batch_idx:
            x_n = np.array(data["x"][n])
            y_n = np.array(data["y"][n])

            pred = [np.dot(w[k], x_n) for k in range(K)]
            pclass = softmax(pred)
            mod_pclass = mod_softmax(pred, C, M)

            A1 = 0
            for k in range(K):
                y_nk = int(y_n == k)
                A1 += y_nk * (pclass[k] / mod_pclass[k])
            A2 = int(y_n == i) / mod_pclass[i]
            gradient[i] += (x_n * pclass[i]) * (A1 - A2)

        gradient[i] = (C * gradient[i]) + (mu * w[i])
    return gradient / B


def FedAvg(policy):
    w_avg = np.random.rand(K, d)
    Queue = np.zeros(K)

    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    client_cnt = np.zeros(N)
    L1sum = 0

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    if policy == "KLGap" or policy == "L2":
        grid = generate_grid()

    eta = 0.01
    decay = 0.9992
    for t in tqdm(range(T)):
        eta = eta * decay
        w_sum = np.zeros((K, d))
        client_loss_train = [
            crossentropy(w_avg, train_data["user_data"][user_name[n]], B, K, C, M, mu)
            for n in range(N)
        ]
        client_loss_test = crossentropy(
            w_avg, test_data["user_data"]["test"], test_size, K, C, M, mu
        )

        client_acc_train = [
            accuracy(w_avg, train_data["user_data"][user_name[n]], B, K, C, M)
            for n in range(N)
        ]
        client_acc_test = accuracy(
            w_avg, test_data["user_data"]["test"], test_size, K, C, M
        )

        loss_train[t] = sum(client_loss_train) / N
        loss_test[t] = client_loss_test
        acc_train[t] = sum(client_acc_train) / N
        acc_test[t] = client_acc_test

        available_client = np.random.choice(N, r, replace=False)

        if policy == "uniform":
            participants_set = client_sample_uni(available_client, m)
        elif policy == "L1":
            participants_set = client_sample_L1(
                train_dist, test_dist, available_client, m
            )
        elif policy == "CBS":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, B, t, client_cnt + 1
            )
        elif policy == "QS":
            participants_set, Queue = client_sample_QS(
                train_dist, test_dist, available_client, m, Queue
            )

        P_mix = np.zeros(K)
        for n in participants_set:
            P_mix = P_mix + train_dist[n, :]
        P_mix = P_mix / m
        L1sum += L1(P_mix, test_dist)

        for n in participants_set:
            client_cnt[n] += 1
            w_n = w_avg[:]
            for i in range(I):
                gradient = gradient_eval(
                    w_n, train_data["user_data"][user_name[n]], C, M
                )
                w_n = w_n - (eta * gradient)
            w_sum = w_sum + w_n
        w_avg = w_sum / m

    return (
        list(loss_train),
        list(loss_test),
        list(acc_train),
        list(acc_test),
        list(client_cnt / T),
        L1sum / T,
    )


def main():
    # Init

    pool = multiprocessing.Pool(initializer=init, processes=4)
    policy_list = ["uniform"]
    results = pool.map(FedAvg, policy_list)

    # TODO: Implement the parallel using multiprocessing module

    (
        loss_train_uni,
        loss_test_uni,
        acc_train_uni,
        acc_test_uni,
        cnt_uni,
        avgL1_uni,
    ) = results[0]

    plt.figure()
    plt.plot(np.linspace(0, T, T), loss_train_uni, "b--", label=r"FedAVg Train Loss")
    plt.plot(np.linspace(0, T, T), loss_test_uni, "b-", label=r"FedAvg Test Loss")
    plt.title(r"Global loss of MNIST dataset with $\alpha = 0.05$ ")
    plt.xlabel("# Communication rounds", fontsize=12)
    plt.ylabel("Global Loss", fontsize=12, rotation=0, ha="right")
    plt.legend(loc="upper right")
    plt.show()

    plt.figure()
    plt.plot(np.linspace(0, T, T), acc_train_uni, "b--", label=r"FedAVg Train Accuracy")
    plt.plot(np.linspace(0, T, T), acc_test_uni, "b-", label=r"FedAvg Test Accuracy")
    plt.title(r"Test Accuracy of MNIST dataset with $\alpha = 0.05$ ")
    plt.xlabel("# Communication rounds", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12, rotation=0, ha="right")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
