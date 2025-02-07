import math
import json
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, repeat
from tqdm import tqdm
import pickle
import sys
from time import time
sys.path.append('code')

from util.cspolicy import (
    client_sample_uni,
    client_sample_QS,
    client_sample_KL,
    client_sample_POC,
    client_sample_CBS,
)
from util.metric import accuracy, crossentropy
from util.misc import softmax, mod_softmax, L1

np.random.seed(seed=12345)

grid = []

N = 100
r = 30
m = 15
d = 28 * 28
K = 10
B = 64
I = 30
T = 10

M = 1000
C = 1 - (K * math.exp(-M))
mu = 1e-3
L = math.sqrt(2) + mu
gamma = (4 * L) / mu
test_size = 600


def import_data(v):
    train_path = f"data/Federated/MNIST_alpha03_{v}_train.pck"
    test_path = f"data/Federated/MNIST_alpha03_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def init(v):
    global train_data
    global test_data
    train_data, test_data = import_data(v)


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
    Aq = np.zeros(N)
    Zq = np.zeros(N)

    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    client_cnt = np.zeros(N)
    class_acc = np.zeros(K)
    client_runtime = np.zeros(T)

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    sumsz_train = sum([train_data["num_samples"][n] for n in range(N)])
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    eta = 0.005
    decay = 0.9992
    for t in tqdm(range(T), miniters=10):
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

        if policy == "POC":
            available_client = np.random.choice(N, r, replace=False, p=szfrac_train)
        else:
            available_client = list(range(N))
        if policy == "uniform":
            participants_set = client_sample_uni(available_client, m)
        elif policy == "CBS":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, B, t, client_cnt + 1
            )
        elif policy == "QS":
            participants_set, Queue = client_sample_QS(
                train_dist, test_dist, available_client, m, Queue
            )
        elif policy == "KL":
            participants_set, Aq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, Aq, Zq
            )
        elif policy == "POC":
            participants_set = client_sample_POC(available_client, m, client_loss_train)
        
        max_time = 0
        gradient_sum = np.zeros((K, d))
        for n in participants_set:
            t_start = time()
            client_cnt[n] += 1
            normalized_gradient = np.zeros((K, d))
            w_n = w_avg[:]
            for i in range(I):
                gradient = gradient_eval(
                    w_n, train_data["user_data"][user_name[n]], C, M
                )
                normalized_gradient += gradient
                w_n = w_n - (eta * gradient)
            normalized_gradient = normalized_gradient / I
            if policy == "CBS" or policy == "L1":
                gradient_sum += normalized_gradient
            else:
                w_sum = w_sum + w_n
            t_end = time()
            max_time = max(t_end-t_start, max_time)
        if policy == "CBS" or policy == "L1":
            w_avg = w_avg - (eta * I * (gradient_sum / m))
        else:
            w_avg = w_sum / m
        client_runtime[t] = max_time

    class_acc = accuracy(
        w_avg, test_data["user_data"]["test"], test_size, K, C, M, perclass=True
    )
    return (
        list(loss_train),
        list(loss_test),
        list(acc_train),
        list(acc_test),
        list(client_cnt / T),
        list(class_acc),
        list(client_runtime)
    )


def main():
    # Init
    run_list = ["v2"]
    for v in run_list:
        print(f"Running version {v}")
        with multiprocessing.Pool(initializer=init, initargs=(v, ), processes=1) as pool:        
            policy_list = ["KL"]
            results = pool.map(FedAvg, policy_list)
            res_dict = {}
            for idx, policy in enumerate(policy_list):
                client_runtime = np.array(results[idx][-1])
                print(np.mean(client_runtime))
            #     res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "acc_class":[]}
            #     for i, metric in enumerate(res_dict[policy]):
            #         res_dict[policy][metric] = results[idx][i]
            # with open(f"results/MNIST_alpha03_{v}.json", "w") as f:
            #     json.dump(res_dict, f)
        
if __name__ == "__main__":
    main()
