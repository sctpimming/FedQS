from math import log, exp
from scipy.special import rel_entr
import numpy as np


def KL(P, Q):
    K = len(P)
    epsilon = 1e-6
    for k in range(K):
        if P[k] == 0:
            P[k] = epsilon
        if Q[k] == 0:
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
