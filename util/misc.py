from math import log, exp
from scipy.special import rel_entr
import numpy as np
from numpy.random import dirichlet, choice

def KL(P, Q):
    K = len(P)
    epsilon = 1e-12
    for k in range(K):
        if P[k] <= 1e-12:
            P[k] = epsilon
        if Q[k] <= 1e-12:
            Q[k] = epsilon
    P = np.array(P)/sum(P)
    Q = np.array(Q)/sum(Q)
    return -sum([P[k] * log(Q[k] / P[k]) for k in range(K)])


def softmax(v):
    v = v - max(v)  # fix overflow problem in softmax, softmax(v) = softmax(v+c)
    expvec = np.exp(v)
    return expvec / sum(expvec)


def mod_softmax(v, C, M):
    return C * softmax(v) + exp(-M)


def L1(P, Q, epsilon=0.01):
    K = len(Q)
    val = 0
    for k in range(K):
        val += (P[k] - Q[k]) ** 2
    return val


def QCID(P, B, K, participant_set, Lb=10 ** (-20)):
    sumval = 0
    # for k in range(K):
    #     nom = 0
    #     denom = 0
    #     for n in participant_set:
    #         nom += B[n]*P[n][k]
    #         denom += B[n]
    #     sumval += ((nom/denom) - (1/K))**2
    E1 = 0
    for n1 in participant_set:
        for n2 in participant_set:
            E1 += B[n1]*B[n2] * np.matmul(P[n1], np.transpose(P[n2])) 
    E2 = sum([B[n] for n in participant_set])**2
    sumval = (E1/E2) - 1/K
    if sumval < Lb:
        print(sumval)
        sumval = Lb
    return sumval

def get_interval(group):
    if group == "G1":
        return (0, 1)
    elif group == "G2":
        return (1, 3)
    elif group == "G3":
        return (3, 5)
    elif group == "G4":
        return (5, 10)
    elif group == "G5":
        return (10, 20)
    elif group == "G6":
        return (20, 30)

def get_distribution(train_group, test_group, N, K):
    # Simulate the noniid-ness using dirichlet distribution
    train_KL = 0
    test_KL = 0
    uni_dist = np.array([1/K]*K)

    train_lb, train_rb = get_interval(train_group)
    test_lb, test_rb = get_interval(test_group)
    magic_coef = 0
    if train_group == "G1":
        magic_coef = 0.5
    elif train_group == "G2":
        magic_coef = 0.3
    elif train_group == "G3":
        magic_coef = 0.1
    elif train_group == "G4":
        magic_coef = 0.05
    else:
        magic_coef = 0.01

    while not(train_KL > train_lb and train_KL <= train_rb):
        alpha_val = (np.random.rand(K)*magic_coef) + magic_coef
        distribution_train = dirichlet(alpha_val, size=N)
        for n in range(N):
            # distribution_train[n] = np.array([0 if v < 1e-12 else v for v in distribution_train[n]])
            distribution_train[n] = np.asarray(distribution_train[n]).astype('float64')
            distribution_train[n] = distribution_train[n]/sum(distribution_train[n])
            distribution_train[n][-1] = 1 - np.sum(distribution_train[n][0:-1])
 
        train_KL = np.mean(np.array([KL(uni_dist, train_dist) for train_dist in distribution_train]))
        print("train dist:", train_KL)

    while not(test_KL > test_lb and test_KL <= test_rb):
        alpha_val = np.random.rand()
        distribution_test = dirichlet([alpha_val] * K, size=1)
        test_KL = KL(uni_dist, distribution_test[0])
        print("test dist:", test_KL)
    return distribution_train, distribution_test

def show_hist(train_dist, test_dist):
    from matplotlib import pyplot as plt
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

def get_partition(data, label_list, K, epsilon=1e-3, lb=1e-6):
    freq = np.zeros(K, dtype=int)
    for k in label_list:
        freq[k] += 1

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
    label_dist = np.ones(K)*epsilon # ensure a non-zero PDF
    for v in y:
        label = v
        label_dist[label] += 1
    label_dist = label_dist / sum(label_dist)
    if len(label_list) > len(y):
        print("Clip alert!: ", len(label_list), len(y))
    if min(label_dist) < lb:
        print("Zero PDF alert the lowest pdf is", min(label_dist))
    return x, y, label_dist, len(y)

def stack(data):
    X = []
    n, d = data.shape
    for i in range(n):
        feature = data[i]
        xsplit = [x.reshape((32, 32)) / 255.0 for x in np.split(feature, 3)]
        rgb = np.dstack((xsplit[0], xsplit[1], xsplit[2]))
        X.append(rgb)
    return X