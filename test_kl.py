import json
from tqdm import tqdm
import numpy as np
from util.cspolicy import client_sample_KL, client_sample_CBS, cvxpy_oneshot
import pickle
from util.misc import KL, QCID
import matplotlib.pyplot as plt
import tenseal as ts
from time import time
from math import comb

import multiprocessing
import concurrent.futures


np.random.seed(12345)

def import_data():
    # train_path = "data/Federated/iNat_train.pck"
    # test_path = "data/Federated/iNat_test.pck"
    train_path = "data/Federated/CIFAR100_alpha03_N100_train.pck"
    test_path = "data/Federated/CIFAR100_alpha03_N100_test.pck"

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def compare_distribution(policy_dist, test_dist):
    # policy_dist = np.matmul(cnt_QS / m, train_dist)
    #test_dist = np.array(test_data["distribution"]["test"])
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

    ylim = max(max(test_dist), max(policy_dist), 0.1)
    for policy in ["QS"]:
        fig, ax = plt.subplots(1, 2)
        if policy == "QS":
            fig.suptitle(f"m = {m}, Distance of FedQS = {KL(test_dist, policy_dist)}")
            ax[0].bar(class_name, policy_dist, color=bar_colors, width=0.9)

        ax[0].set_xlabel("Label")
        ax[0].set_ylabel("pmf", rotation=0, ha="right")
        ax[0].set_ylim([0, ylim])

        ax[1].set_xlabel("Label")
        ax[1].set_ylabel("pmf", rotation=0, ha="right")
        ax[1].set_ylim([0, ylim])
        ax[1].bar(class_name, test_dist, color=bar_colors, width=0.9)
        plt.show()

def plot_distribution(client_rate, policy_list, m):
    # K =1203
    first_key = policy_list[0]
    train_data, test_data = import_data()
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    class_list = np.array(list(range(K)))
    class_name = [f"{n}" for n in class_list]
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

    support_list = np.array([test_dist[k] for k in class_list])
    top = 25
    top_index = np.argsort(support_list)[-top:][::-1]
    bottom_index = np.argsort(support_list)[:top]
    print(top_index)
    major_class = class_list[top_index]
    minor_class = class_list[bottom_index]

    values_top = [[test_dist[k] for k in major_class]]
    values_bottom = [[test_dist[k] for k in minor_class]]
    group_labels = ["Test distribution"]

    fig, ax = plt.subplots()

    for policy in policy_list:
        group_labels.append(policy)
        policy_dist = np.matmul(client_rate, train_dist)
        print(f"{policy} divergence is {KL(test_dist, policy_dist)}")
        if "KL" or "QS" in policy:
            fig.suptitle(f"m = {m}, Distance of FedQS = {KL(test_dist, policy_dist)}")
        major_val = [policy_dist[k] for k in major_class]
        minor_val = [policy_dist[k] for k in minor_class]

        values_top.append(major_val)
        values_bottom.append(minor_val)
    
    
    categories = [f"{k}" for k in major_class]
    n_categories = len(categories)
    n_groups = len(group_labels)

    index = np.arange(n_categories)
    bar_width = 0.2

    for i in range(n_groups):
        plt.bar(index + i * bar_width, values_top[i], bar_width, label=group_labels[i], color=bar_colors[i])

    ax.set_ylabel('PMF')
    ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_title(f"top-{top} major classes distribution")
    ax.legend()

    plt.show()

def plot_shift(shift_list):
    T = len(shift_list)
    plt.plot(np.arange(T), shift_list, label="Distribution shift")
    # plt.plot(np.arange(T), QCID_list, label="QCID")
    plt.xlabel("Number of rounds")
    plt.ylabel("Distribution Shift")
    plt.show()

def csprofile(client_cnt):
    plt.figure()
    binwidth = 0.005
    data = client_cnt
    print(len([x for x in data if x > 0]))
    density, bins, _ = plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    count, _ = np.histogram(data, bins)
    for x,y,num in zip(bins, density, count):
        if num != 0:
            plt.text(x, y+0.05, num, fontsize=10) # x,y,str
    # plt.bar_label(bars)
    plt.xlabel("Participation rate", fontsize=10)
    plt.xticks(np.arange(min(data), max(data) + binwidth*5, binwidth*5))
    plt.ylabel("Client count", fontsize=10)

    plt.show()

def solve_oneshot(m_val = 500, V_val = 100):
    sample_prob = cvxpy_oneshot(
        train_dist, test_dist, m_val, nsample_train, V_val, Q_inv
    )
    client_rate = np.array(sample_prob)/m_val
    policy_dist = np.matmul(client_rate, train_dist)
    print(sum(policy_dist))
    plot_distribution(client_rate, ["QS_oneshot"], m_val)
    return client_rate

def matching(m_val=5, V_val=50, T=500000):
    print(f"Running M = {m_val}, V = {V_val}")
    oneshot_feasible = (comb(N, m_val) < 100000)
    if oneshot_feasible:
        oneshot_rate = solve_oneshot(m_val, V_val)
    Aq = np.zeros(N)
    Yq = np.zeros(N)
    Zq = np.zeros(N)
    client_cnt = np.zeros(N)

    shift_list = np.zeros(T)
    Aq_size_list = np.zeros(T)
    solution_dist = np.zeros(T)

    for t in tqdm(range(T)):
        Aq_size_list[t] = np.linalg.norm(Aq)
        participants_set, Aq, Yq, Zq = client_sample_KL(
            train_dist, test_dist, available_client, m_val, nsample_train, 
            Aq, Yq, Zq, Q_inv, V=V_val, R=(m_val/N)*0, max_rate_lim=False
        )
        for n in participants_set:
            client_cnt[n] += 1
        client_rate = client_cnt/((t+1)*m_val)
        policy_dist = np.matmul(client_rate, train_dist)
        shift_list[t] = KL(test_dist, policy_dist)
        print(f"M = {m_val}, V = {V_val} iter {t}: shift = {shift_list[t]:.4f} Aq_size = {Aq_size_list[t]:.4f}")
        if oneshot_feasible:
            solution_dist[t] = np.linalg.norm(client_rate-oneshot_rate)
            print(f"solution distance = {solution_dist[t]:.4f}")
    client_rate = client_cnt / T
    res_dict = {
        "policy_dist":list(policy_dist), 
        "test_dist":list(test_dist),
        "client_rate":list(client_rate),
        "shift_list":list(shift_list),
        "Aq_size":list(Aq_size_list),
        "sol_dist":list(solution_dist),
        "M":m_val,
        "V":V_val
    }
    return res_dict

def sweep_over_m(V_val = 50):
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("fork")) as executor:
            futures = [executor.submit(matching, m_val, V_val) for m_val in m_list]
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                m_val = results.pop("M")
                V_val = results.pop("V")
                print(f"m = {m_val} and V = {V_val} is completed.")
                with open(f"results/CIFAR100_testKL_M{m_val}_V{V_val}_N100.json", "w") as f:
                    json.dump(results, f)  
            

def sweep_over_V(m_val=5):
    with concurrent.futures.ProcessPoolExecutor(max_workers=9, mp_context=multiprocessing.get_context("fork")) as executor:
            futures = [executor.submit(matching, m_val, V_val) for V_val in V_list]
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                m_val = results.pop("M")
                V_val = results.pop("V")
                print(f"m = {m_val} and V = {V_val} is completed.")
                with open(f"results/CIFAR100_testKL_M{m_val}_V{V_val}_N100.json", "w") as f:
                    json.dump(results, f)  


def FHE_runtime(train_dist, test_dist):
    # Setup TenSEAL context
    context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
    context.generate_galois_keys()
    context.global_scale = 2**40

    N, K = train_dist.shape
    # N = 1000
    m = 15
    # K = 100
    T = 20
    max_itr = 20 # max_itr for convex solver
    available_client = list(range(N))
    Aq = np.zeros(N)
    Zq = np.zeros(N)

    solve_time = np.zeros(T)
    v = np.random.rand(N, K)
    w = 1/N * np.ones(N)
    v_enc = []

    # Client send encrypted label distribution
    for n in range(N):
        v_enc.append(ts.ckks_vector(context, v[n, :]))

    for t in tqdm(range(T)):

        # Server Processes
        t_start = time()

        v_mix = ts.ckks_vector(context, np.zeros(K))
        for itr in range(max_itr):
            for n in range(N): # Calculate the distribution mixture
                v_mix = v_mix + (w[n] * v_enc[n])
            # communicate to the oracle
            # calculate the objective value
            # communicate back to the selector
            for n in range(N): # Update the mixture weight
                w[n] -= (np.random.rand()*2) - 1
            
        participants_set, Aq, Zq = client_sample_KL(
            train_dist, test_dist, available_client, m, Aq, Zq
        )
        t_end = time()
        solve_time[t] = (t_end-t_start)
    print(solve_time, np.mean(solve_time))

    

N = 100
# m = 100
K = 100
B = 50
# N = 100
# m = 15
# K =100

train_data, test_data = import_data()
train_dist = np.array(
    [train_data["distribution"][uname] for uname in train_data["distribution"]]
)
nsample_train = [train_data["num_samples"][n] for n in range(N)]
test_dist = np.array(test_data["distribution"]["test"])

available_client = list(range(N))
uniform_dist = sum(train_dist)/N
print(uniform_dist.shape)
print(f"Uniform Shift is: {KL(test_dist, uniform_dist)}")
# test_dist = [1/K]*K

Q_inv = np.array([1/q if q > 0 else 10 ** (-20) for q in test_dist])
m_list = [1, 2]
# m_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
V_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]

# V_list = [200, 500]
# V_list = [5, 500]

# sweep_over_m(V_val=100)
# sweep_over_V(m_val=2)
solve_oneshot(m_val=2, V_val=500)

# FHE_runtime(train_dist, test_dist)
