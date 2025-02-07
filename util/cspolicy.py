import numpy as np
import math
from util.misc import KL, QCID
from cvxopt import solvers, matrix, spdiag, log, div, mul
from cvxopt.modeling import dot
from itertools import combinations
from tqdm import tqdm
from math import comb
import cvxpy as cp

def obj_solve(P, Q, m, B, Aq, V, Q_inv):
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
        prob.solve(verbose=False, solver=cp.SCS, canon_backend=cp.SCIPY_CANON_BACKEND, max_iters=1000)
    except Exception as e:
        print(e)
        print("Using the uniform sampling")
        return [m/N for n in range(N)]
    return [x[n].value for n in range(N)]

def diminish_obj_solve(P, Q, m, B, Aq, V):
    N, K = P.shape
    x = cp.Variable(N)
    P_mix = ((cp.power(x, 1/16))@P)/m
    obj = cp.Minimize(
        (-V * cp.sum(cp.multiply(Q, cp.log(P_mix))))
        + cp.sum(cp.multiply(Aq, x))
    )
    constraints = [x >= 0, -x >= -1]

    prob = cp.Problem(obj, constraints)

    
    try:
        prob.solve(verbose=False, solver=cp.SCS, canon_backend=cp.SCIPY_CANON_BACKEND)
    except Exception as e:
        print(e)
        print("Using the uniform sampling")
        return [m/N for n in range(N)]
    return [x[n].value for n in range(N)]

def cvxpy_oneshot(P, Q, m, B, V, Q_inv):
    N, K = P.shape
    x = cp.Variable(comb(N, m))
    comb_list = combinations(list(range(N)), m)
    comb_mat = np.zeros((comb(N, m), N))
    for i, val in enumerate(comb_list):
        one_hot = np.zeros(N)
        for j in val:
            one_hot[j] = 1
        comb_mat[i] = one_hot
    rate = x@comb_mat
    P_mix = rate@P
    obj = cp.Minimize(
        -cp.sum(cp.multiply(Q, cp.log(cp.multiply(P_mix, Q_inv))))
    )
    constraints = [rate >= 0, -rate >= -1, cp.sum(rate) == m]
    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(verbose=True, solver=cp.SCS)
    except Exception as e:
        print(e)
        print("Using the uniform sampling")
        return [m/N for n in range(N)]
    return list(x.value@comb_mat)
def subproblem_solve(N, m, Aq, V):
    x = cp.Variable(N)
    # B = np.array([B]*N)
    # print(B.size, cp.min(x, B))
    obj = cp.Maximize(
        (V * cp.sum(cp.log(x))) - cp.sum(cp.multiply(Aq, x))
    )
    constraints = [x >= 0, -x >= -1]

    prob = cp.Problem(obj, constraints)
    
    try:
        prob.solve(verbose=False, solver=cp.SCS, canon_backend=cp.SCIPY_CANON_BACKEND, max_iters=1000)
    except Exception as e:
        print(e)
        print("Using the uniform sampling")
        return [m/N for n in range(N)]
    return [x[n].value for n in range(N)]

def client_sample_uni(R, m):
    participants_set = np.random.choice(R, m, replace=False)
    return participants_set

def client_sample_POC(R, m, client_loss):
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

def client_sample_CBS(P, R, m, B, t, client_cnt, expfactor=10):
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
def client_sample_comb(P, Q, m, Aq, Bq, V, R):
    N, K = P.shape
    aux = list(subproblem_solve(N, m, Aq, V))
    comb_list = combinations(list(range(N)), m)
    participant_set = []
    best_val = np.inf
    shift_val = 0
    for cand_sol in comb_list:
        one_hot = np.array([1 if n in cand_sol else 0 for n in range(N)])
        P_mix = np.matmul(one_hot, P)/m
        obj_val = Bq*KL(Q, P_mix) - sum(np.multiply(Aq, one_hot))
        if obj_val < best_val:
            participants_set = cand_sol
            best_val = obj_val
            shift_val = KL(Q, P_mix)

    for n in range(N):
        if n in participants_set:
            Aq[n] = max(Aq[n] + aux[n] - 1, 0)
        else:
            Aq[n] = max(Aq[n] + aux[n], 0)
    Bq = max(Bq + shift_val - R, 0)

    return participants_set, Aq, Bq

def client_sample_KL(P, Q, A, m, B, Aq, Yq, Zq, Q_inv, V=5, R=0, max_rate_lim=False, diminish=False):
    N, K = P.shape
    if not diminish:
        aux = list(obj_solve(P, Q, m, B, Aq, V, Q_inv))
    else:
        max_rate_lim = False
        aux = list(diminish_obj_solve(P, Q, m, B, Aq, V))

    
    Rmax = R*5
    participants_set = []
    queue_val = {}
    for n in A:
        queue_val[n] = Aq[n] - Yq[n] + Zq[n]
    sorted_val = dict(sorted(queue_val.items(), key=lambda item: item[1], reverse=True))
    sorted_key = list(sorted_val.keys())
    participants_set = sorted_key[:m]

    for n in sorted_key:
        if n in participants_set:
            Aq[n] = max(Aq[n] + aux[n] - 1, 0)
            Zq[n] = max(Zq[n] + R - 1, 0)
            if max_rate_lim == True:
                Yq[n] = max(Yq[n] + 1 - Rmax, 0) 
        else:
            Aq[n] = max(Aq[n] + aux[n], 0)
            Zq[n] = max(Zq[n] + R, 0)
            if max_rate_lim == True:
                Yq[n] = max(Yq[n] - Rmax, 0)
    return participants_set, Aq, Yq, Zq