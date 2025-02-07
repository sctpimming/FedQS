import json
from tqdm import tqdm
import numpy as np
from util.cspolicy import client_sample_KL, client_sample_CBS
import pickle
from util.misc import KL, QCID
import matplotlib.pyplot as plt
import tenseal as ts
from time import time

def import_data():
    # train_path = "data/Federated/iNat_train.pck"
    # test_path = "data/Federated/iNat_test.pck"
    train_path = "data/Federated/CIFAR100_alpha03_v6_train.pck"
    test_path = "data/Federated/CIFAR100_alpha03_v6_test.pck"

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def plot_m_vs_shift(V_val=50):
    m_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    res_dict = {}
    for m in m_list:
        with open(f"results/CIFAR100_testKL_M{m}_V{V_val}_hull.json", "r") as infile:
            result = json.load(infile)
            res_dict[m] = {"policy_dist":[], "test_dist":[],"client_rate":[]}
            for i, metric in enumerate(res_dict[m]):
                res_dict[m][metric] = result[metric]

    policy_dist_list = [KL(res_dict[m]["test_dist"], res_dict[m]["policy_dist"]) for m in m_list]
    plt.plot(m_list, policy_dist_list, "-o")
    plt.xlabel("M")
    plt.xscale("log")
    plt.ylabel("Shift")
    plt.show()

def plot_V_vs_shift(m=5):
    V_list = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    res_dict = {}
    for V in V_list:
        with open(f"results/CIFAR100_testKL_M{m}_V{V}_N100.json", "r") as infile:
            result = json.load(infile)
            res_dict[V] = {"policy_dist":[], "test_dist":[],"client_rate":[]}
            for i, metric in enumerate(res_dict[V]):
                    res_dict[V][metric] = result[metric]
    policy_dist_list = [KL(res_dict[V]["test_dist"], res_dict[V]["policy_dist"]) for V in V_list]
    plt.axhline(y = 0.0328, color = 'r', linestyle = 'dashed', label="Oneshot solution")  
    plt.plot(V_list, policy_dist_list, "-o", label = "DPP solutions")
    plt.xlabel("V")
    plt.xscale("log")
    plt.ylabel("Shift")
    plt.title(f"m = {m}")
    plt.legend()
    plt.show()

def plot_shift(m, V):
    res_dict = {}
    with open(f"results/CIFAR100_testKL_M{m}_V{V}_N100.json", "r") as infile:
            result = json.load(infile)
            res_dict[V] = {"policy_dist":[], "test_dist":[],"client_rate":[], "shift_list":[]}
            for i, metric in enumerate(res_dict[V]):
                    res_dict[V][metric] = result[metric]
    # policy_dist_list = [KL(res_dict[V]["test_dist"], res_dict[V]["policy_dist"]) for V in V_list]
    T = len(res_dict[V]["shift_list"])
    plt.plot(np.arange(T), res_dict[V]["shift_list"])
    plt.xlabel("# of rounds")
    plt.ylabel("Shift")
    plt.show()
def plot_backlog(m, V):
    res_dict = {}
    with open(f"results/CIFAR100_testKL_M{m}_V{V}_hull.json", "r") as infile:
            result = json.load(infile)
            res_dict[V] = {"policy_dist":[], "test_dist":[],"client_rate":[], "shift_list":[], "Aq_size":[]}
            for i, metric in enumerate(res_dict[V]):
                    res_dict[V][metric] = result[metric]
    # policy_dist_list = [KL(res_dict[V]["test_dist"], res_dict[V]["policy_dist"]) for V in V_list]
    T = len(res_dict[V]["shift_list"])
    for t in range(T):
         res_dict[V]["Aq_size"][t] = res_dict[V]["Aq_size"][t]/(t+1)
    print(min(res_dict[V]["Aq_size"][1:]))
    plt.plot(np.arange(T), res_dict[V]["Aq_size"])
    plt.xlabel("# of rounds")
    plt.ylabel("|Aq|/t")
    # plt.yscale("log")
    plt.show()
def plot_solution_dist(m, V):
    res_dict = {}
    with open(f"results/CIFAR100_testKL_M{m}_V{V}_N100.json", "r") as infile:
            result = json.load(infile)
            res_dict[V] = {"policy_dist":[], "test_dist":[],"client_rate":[], "shift_list":[], "Aq_size":[], "sol_dist":[]}
            for i, metric in enumerate(res_dict[V]):
                    res_dict[V][metric] = result[metric]
    # policy_dist_list = [KL(res_dict[V]["test_dist"], res_dict[V]["policy_dist"]) for V in V_list]
    T = len(res_dict[V]["shift_list"])
    plt.plot(np.arange(T), res_dict[V]["sol_dist"])
    plt.xlabel("# of rounds")
    plt.ylabel("L2 from solution")
    # plt.yscale("log")
    plt.show()
     
     
N=5000
K=100
B=50

# plot_m_vs_shift()
plot_V_vs_shift(m=2)
# plot_shift(m=1, V=500)
# plot_solution_dist(m=1, V=500)
# plot_backlog(m=5, V=50)


    
    

