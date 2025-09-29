import numpy as np
import math
from util.misc import KL, QCID
from cvxopt import solvers, matrix, spdiag, log, div, mul
from cvxopt.modeling import dot
from itertools import combinations
from tqdm import tqdm
from math import comb
from scipy.optimize import nnls
import cvxpy as cp

def softmax(x):
    return np.array(x) / sum(x)

def obj_solve(P, Q, m, Aq, V):
    N, K = P.shape
    x = cp.Variable(N)
    P_mix = (x@P)/m
    
    obj = cp.Minimize(
        (-V * cp.sum(cp.multiply(Q, cp.log(P_mix))))
        + cp.sum(cp.multiply(Aq, x))
    )
    constraints = [x >= 0, -x >= -1]

    prob = cp.Problem(obj, constraints)

    
    try:
        prob.solve(verbose=False, solver=cp.SCS, canon_backend=cp.SCIPY_CANON_BACKEND, eps=1e-3)
    except Exception as e:
        print(e)
        print("Using the uniform sampling")
        return [m/N for n in range(N)]
    return [x[n].value for n in range(N)]

def client_sample_uni(metric):
    R = metric["available_client"]
    m = metric["n_participants"]
    n_samples = [metric["client_sample"][r] for r in R]
    fraction_of_samples = np.array(n_samples)/sum(n_samples)
    participants_set = np.random.choice(R, m, replace=False, p=fraction_of_samples)
    return participants_set

def client_sample_POC(metric, dmult):
    R = metric["available_client"]
    m = metric["n_participants"]
    client_loss = metric["client_loss"]

    n_samples = [metric["client_sample"][r] for r in R]
    fraction_of_samples = np.array(n_samples)/sum(n_samples)
    R = np.random.choice(R, m*dmult, replace=False, p=fraction_of_samples) 

    participants_set = []
    queue_val = {}
    for n in R:
        queue_val[n] = client_loss[n]
    sorted_val = dict(sorted(queue_val.items(), key=lambda item: item[1], reverse=True))
    for n in sorted_val.keys():
        participants_set.append(n)
        if len(participants_set) == m:
            break
    return participants_set

def client_sample_CBS(metric, expfactor=10):
    P = metric["train_dist"]
    R = metric["available_client"]
    m = metric["n_participants"]
    B = metric["client_sample"]
    t = metric["round"]
    client_cnt = metric["client_count"]

    N, K = P.shape
    participants_set = []
    psample1 = [
        1 / (QCID(P, B, K, participants_set + [n]))
        + (expfactor * np.sqrt((3 * log(t + 1)) / (2 * client_cnt[n])))
        for n in R
    ]
    client_idx = np.random.choice(R, p=psample1 / sum(psample1))
    participants_set.append(client_idx)

    psample2_candidate = [1 / (QCID(P, B, K, participants_set + [n]) ** 2) for n in R]
    psample2 = np.divide(psample2_candidate, psample1)
    client_idx = np.random.choice(R, p=psample2 / sum(psample2))
    participants_set.append(client_idx)

    QCID_old = QCID(P, B, K, participants_set)
    for idx in range(3, m + 1):
        beta = idx
        QCID_candidate = [QCID(P, B, K, participants_set + [n]) ** beta for n in R]
        psample = np.divide(QCID_old ** (beta - 1), QCID_candidate)
        client_idx = np.random.choice(R, p=psample / sum(psample))
        participants_set.append(client_idx)
        QCID_old = QCID(P, B, K, participants_set)

    return participants_set

def client_sample_ODFL(metric):
    
    P = metric["train_dist"]
    target_dist = metric["test_dist"]
    num_sampled_clients = metric["n_participants"]
    available_clients = metric["available_client"]

    n_ready = len(available_clients)

    N, K = P.shape
    subset_size = num_sampled_clients
    client_index = {}

    selected_client_subset = []
    recursive_weight = [0] * N
    client_dist = np.zeros((n_ready, K))

    max_i = 0
    idx = 0
    for n in available_clients:
        client_dist[idx] = P[n]
        client_index[idx] = n
        idx += 1


    try:
        while (len(selected_client_subset) < subset_size):
            max_i += 1
            preds_w, preds_err = nnls(client_dist.T, target_dist)
 
            for i in np.where(preds_w != 0)[0]:
                real_index = client_index[i]
                selected_client_subset.append(real_index)
                recursive_weight[real_index] += preds_w[i]
                client_dist[i] = np.array([0]*client_dist[i].shape[0])


        recursive_weight = softmax(np.array(recursive_weight)).flatten()
        client_subset = np.where(recursive_weight != 0)[0]
        recursive_weight_subset = [recursive_weight[n] for n in client_subset]

        sampled_client_indices = np.random.choice(client_subset, size=num_sampled_clients, replace=False, p = recursive_weight_subset).tolist()
    except Exception as e:
        print(e)
        sampled_client_indices = np.random.choice(available_clients, size=num_sampled_clients, replace=False).tolist()

    return sampled_client_indices, recursive_weight

def client_sample_KL(metric, V=5, min_rate_lim=False, max_rate_lim=False):
    P = metric["train_dist"]
    Q = metric["test_dist"]
    A = metric["available_client"]
    m = metric["n_participants"]
    Aq, Yq, Zq = metric["queue_backlog"]

    N, K = P.shape

    n_samples = [metric["client_sample"][n] for n in range(N)]
    fraction_of_samples = np.array(n_samples)/sum(n_samples)
    client_weight = fraction_of_samples

    
    aux = list(obj_solve(P, Q, m, Aq, V))
    R = [m * client_weight[n] for n in range(N)]
    Rmax = R*10
    participants_set = []
    queue_val = {}
    for n in A:
        queue_val[n] = Aq[n] - Yq[n] + Zq[n]
    sorted_val = dict(sorted(queue_val.items(), key=lambda item: item[1], reverse=True))
    sorted_key = list(sorted_val.keys())
    participants_set = sorted_key[:m]

    for n in range(N):
        if n in participants_set:
            Aq[n] = max(Aq[n] + aux[n] - 1, 0)
            if min_rate_lim == True:
                Zq[n] = max(Zq[n] + R[n] - 1, 0)
            if max_rate_lim == True:
                Yq[n] = max(Yq[n] + 1 - Rmax[n], 0) 
        else:
            Aq[n] = max(Aq[n] + aux[n], 0)
            if min_rate_lim == True:
                Zq[n] = max(Zq[n] + R[n], 0)
            if max_rate_lim == True:
                Yq[n] = max(Yq[n] - Rmax[n], 0)
    return participants_set, (Aq, Yq, Zq)