import json
import numpy as np
import matplotlib.pyplot as plt
from util.misc import KL
import pickle

N = 5000
m = 5

dataset = "CIFAR100"
alpha = "alpha01"
version = "v1"
target_acc = 0.30

def import_data():
    train_path = f"data/Federated/{dataset}_{alpha}_{version}_train.pck"
    test_path = f"data/Federated/{dataset}_{alpha}_{version}_test.pck"
    # train_path = f"data/Federated/iNat_train.pck"
    # test_path = f"data/Federated/iNat_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def plot_distribution(results, Debug=False, UseName=True):
    K =100
    if UseName is True:
        train_data, test_data = import_data()
        train_dist = np.array(
            [train_data["distribution"][uname] for uname in train_data["distribution"]]
        )
        test_dist = np.array(test_data["distribution"]["test"])
    else:
        train_dist = results["uniform"]["train_dist"]
        test_dist = results["uniform"]["test_dist"]
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

    policy_list = ["V1", "V10", "V100", "V500"]
    label_list = ["V1", "V10", "V100", "V500"]
    ylim = max(test_dist)
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    # for outidx, subfig in enumerate(subfigs.flat):
    subfigs[0].suptitle(f'Time-average label distribution of selected clients')
    axs = subfigs[0].subplots(2, 2)
    plt.subplots_adjust(top=0.89,
                        bottom=0.125,
                        left=0.125,
                        right=0.9,
                        hspace=0.25,
                        wspace=0.25)
    for inidx, ax in enumerate(axs.flat):
        policy = policy_list[inidx]
        client_cnt = np.array(results[policy]["client_cnt"])/m
        policy_dist = np.matmul(client_cnt, train_dist)
        print(KL(test_dist, policy_dist))
        ax.bar(class_name, policy_dist, color=bar_colors, width=0.9)
        ax.title.set_text(f"{label_list[inidx]}")
        ax.set_ylim([0, ylim])
        ax.set_ylabel("PMF")
        ax.set_xlabel("Label")
    
    ax = subfigs[1].subplots(1, 1)
    ax.bar(class_name, test_dist, color=bar_colors, width=0.9)
    ax.title.set_text("Testing label distribution")
    ax.set_ylim([0, ylim])
    ax.set_ylabel("PMF")
    ax.set_xlabel("Label")

    plt.show()
def plot_acc(
    results,
    policy_list,
    showtrain=False,
    Debug=False
):
    first_key = policy_list[0]
    T = len(results[first_key]["acc_train"])
    npolicy = len(policy_list)
    # policy_list = ["uniform", "POC", "CBS", "KL"]
    # policy_list = ["POC", "CBS", "KL"]
    style_list = ["b-", "r-", "y-", "g-", "m-", "c-", "k-"]
    label_list = policy_list
    # label_list = ["V=1", "V=5", "V=10", "V=50", "V=100", "V=500"]
    # label_list = ["Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    linew = 0.8
    plt.figure(figsize=(3, 3))
    plt.grid()
    
    for idx, policy in enumerate(policy_list):
        plt.plot(
            np.linspace(0, T, T), results[policy]["acc_test"], style_list[idx], linewidth=linew, label=label_list[idx]
        )


    if showtrain == True:
        for idx, policy in enumerate(policy_list):
            plt.plot(
                np.linspace(0, T, T), results[policy]["acc_train"], style_list[idx]+"-", linewidth=linew, label="Train " + label_list[idx]
            )
    
    if Debug == True:
        for idx, policy in enumerate(policy_list):
            print(f"{label_list[idx]} best test accuracy: {max(results[policy]['acc_test'])}")
            for itr, v in enumerate(results[policy]["acc_test"]):
                if v >= target_acc:
                    print(f"{label_list[idx]} takes {itr} iterations to reach {target_acc} accuracy")
                    break

    # plt.title(r"Test Accuracy of MNIST dataset")
    plt.xlabel("Number of rounds", fontsize=10)
    plt.ylabel("Testing accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.show()
def plot_runtime():
    time_dict = {"MNIST": [0.9323, 4.7180, 9.4093],
                "CIFAR10": [0.9208, 4.6502, 9.8389],
                "Fashion-MNIST": [0.9235, 4.6400, 9.3233],
                "CIFAR100":[1.4655, 7.1860, 14.4594]
                }
    N_list = [100, 500, 1000]
    plt.plot(N_list, time_dict["MNIST"], marker="o", label="MNIST")
    plt.plot(N_list, time_dict["CIFAR10"], marker="o", label="CIFAR10")
    plt.plot(N_list, time_dict["Fashion-MNIST"], marker="o", label="Fashion-MNIST")
    plt.plot(N_list, time_dict["CIFAR100"], marker="o", label="CIFAR100")
    plt.xticks(N_list)
    plt.xlabel("Number of clients")
    plt.ylabel("Selector runtime (seconds)")
    plt.legend()
    plt.show()

def plot_csprofile(client_cnt):
    plt.figure()
    binwidth = 0.0005
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

def plot_overall(res_dict, V_val):
    train_data, test_data = import_data()
    train_dist = np.array(
            [train_data["distribution"][uname] for uname in train_data["distribution"]]
        )
    test_dist = np.array(test_data["distribution"]["test"])
    policy_list = [f"V{val}" for val in V_val]
    max_acc_list = []
    policy_shift_list = []
    target_acc_list = []
    T = 2000
    target_acc = 0.15
    for policy in policy_list:
        client_rate = np.array(res_dict[policy]["client_cnt"])/m
        policy_dist = np.matmul(client_rate, train_dist)
        policy_shift = KL(test_dist, policy_dist)
        max_acc = max(res_dict[policy]['acc_test'])
        max_acc_list.append(max_acc)
        policy_shift_list.append(policy_shift)
        for itr, v in enumerate(res_dict[policy]["acc_test"][:T]):
            if v >= target_acc:
                target_acc_list.append(itr)
                break
    
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(V_val, max_acc_list, 'b-o')
    ax2.plot(V_val, policy_shift_list, 'r-o')
    ax1.set_xlabel("V")
    ax1.set_ylabel("accuracy")
    ax1.set_xscale("log")
    ax1.set_xticks(V_val)

    ax1.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')

    ax2.set_ylabel("Distribution shift")
    plt.show()

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(V_val, target_acc_list, 'b-o')
    ax2.plot(V_val, policy_shift_list, 'r-o')
    ax1.set_xlabel("V")
    ax1.set_ylabel("# rounds to reach target acc")
    ax1.set_xscale("log")
    ax1.set_xticks(V_val)

    ax1.yaxis.label.set_color('blue')
    ax2.yaxis.label.set_color('red')

    ax2.set_ylabel("Distribution shift")
    plt.show()




def main():
    print(f"Visualize results from dataset {dataset} with alpha = {alpha} from version {version}")
    # policy_list = ["V10", "V20", "V50", "V100", "V200", "V500"]
    # policy_list = ["V0.1", "V0.2", "V0.5", "V1", "V2", "V5"]
    V_val = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    policy_list = [f"V{val}" for val in V_val]

    res_dict = {}

    for policy in policy_list:
        with open(f"results/{dataset}_{policy}_R1_hull.json", "r") as infile:
            result = json.load(infile)
            result = result[f"{policy[1:]}"]
            res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
            for i, metric in enumerate(res_dict[policy]):
                res_dict[policy][metric] = result[metric]
    # for policy in policy_list:
    #     with open(f"results/{dataset}_{alpha}_S4_{policy}.json", "r") as infile:
    #         result = json.load(infile)
    #         res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
    #         for i, metric in enumerate(res_dict[policy]):
    #             res_dict[policy][metric] = result[metric]
    
    
    # plot_acc(res_dict, policy_list, showtrain=False, Debug=True)
    # plot_csprofile(res_dict, policy_list)
    # plot_distribution(res_dict, Debug=True, UseName=True)
    plot_csprofile(res_dict["V1"]["client_cnt"])
    plot_overall(res_dict, V_val)


if __name__ == "__main__":
    main()