import json
import numpy as np
import matplotlib.pyplot as plt
from util.misc import KL
import pickle

def import_data():
    train_path = f"data/Federated/{dataset}_{alpha}_{version}_train.pck"
    test_path = f"data/Federated/{dataset}_{alpha}_{version}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def plot_distribution(results, policy_list):
    # K =1203
    first_key = policy_list[0]
    T = len(results[first_key]["acc_test"])
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

    for policy in policy_list:
        group_labels.append(policy)
        client_rate = np.array(results[policy]["client_cnt"])/m
        policy_dist = np.matmul(client_rate, train_dist)
        print(f"{policy} divergence is {KL(test_dist, policy_dist)}")
        major_val = [policy_dist[k] for k in major_class]
        minor_val = [policy_dist[k] for k in minor_class]

        values_top.append(major_val)
        values_bottom.append(minor_val)
    
    
    categories = [f"{k}" for k in major_class]
    n_categories = len(categories)
    n_groups = len(group_labels)

    index = np.arange(n_categories)
    bar_width = 0.2

    fig, ax = plt.subplots()
    for i in range(n_groups):
        plt.bar(index + i * bar_width, values_top[i], bar_width, label=group_labels[i], color=bar_colors[i])

    ax.set_ylabel('PMF')
    ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_title(f"top-{top} major classes distribution")
    ax.legend()

    plt.show()

def plot_loss(
    results,
    showtrain=False,
):

    linew = 0.8
    plt.figure(figsize=(3, 3))
    plt.grid()
    
    for idx, policy in enumerate(policy_list):
        plt.plot(
            np.linspace(0, T, T), results[policy]["loss_test"], style_list[idx], linewidth=linew, label=label_list[idx]
        )


    if showtrain == True:
        for idx, policy in enumerate(policy_list):
            plt.plot(
                np.linspace(0, T, T), results[policy]["loss_train"], style_list[idx]+"-", linewidth=linew, label="Train " + label_list[idx]
            )

    plt.xlabel("# Communication rounds", fontsize=12)
    plt.ylabel("Global Loss", fontsize=12, ha="right")
    plt.legend(loc="upper right")
    plt.show()

def plot_acc(
    results,
    policy_list,
    showtrain=False,
    Debug=False
):
    T = len(results[policy_list[0]]["acc_train"])
    target_acc = 0.3
    npolicy = len(policy_list)
    style_list = ["b-", "r-", "y-", "g-", "m-"]
    label_list = policy_list
    linew = 0.8
    plt.figure(figsize=(10, 6))
    plt.grid()
    for idx, policy in enumerate(policy_list):
        plt.plot(
            np.linspace(0, T, T), results[policy]["acc_test"][:T], style_list[idx], linewidth=linew, label=label_list[idx]
        )


    if showtrain == True:
        for idx, policy in enumerate(policy_list):
            plt.plot(
                np.linspace(0, T, T), results[policy]["acc_train"], style_list[idx]+"-", linewidth=linew, label="Train " + label_list[idx]
            )
    
    if Debug == True:
        for idx, policy in enumerate(policy_list):
            print(f"{label_list[idx]} best test accuracy: {max(results[policy]['acc_test'][:T])}")
            # print(f'Rare Client Selection Rate {results[policy]["client_cnt"][:3]}')
            for itr, v in enumerate(results[policy]["acc_test"][:T]):
                if v >= target_acc:
                    print(f"{label_list[idx]} takes {itr} iterations to reach {target_acc} accuracy")
                    break

    plt.xlabel("Number of rounds", fontsize=10)
    plt.ylabel("Testing accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.show()

def plot_f1(
    results,
    policy_list,
):
    T = len(results[policy_list[0]]["acc_train"])
    npolicy = len(policy_list)
    # policy_list = ["uniform", "POC", "CBS", "KL"]
    # policy_list = ["POC", "CBS", "KL"]
    style_list = ["b-", "r-", "y-", "g-", "m-"]
    # label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    label_list = policy_list
    linew = 0.8
    plt.figure(figsize=(3, 3))
    plt.grid()
    
    for idx, policy in enumerate(policy_list):
        print(f"{label_list[idx]} best test f1 score: {max(results[policy]['f1_test'])}")
        plt.plot(
            np.linspace(0, T, T), results[policy]["f1_test"], style_list[idx], linewidth=linew, label=label_list[idx]
        )
        

    # plt.title(r"Test Accuracy of MNIST dataset")
    plt.xlabel("Communication rounds", fontsize=10)
    plt.ylabel("F1 score", fontsize=10)
    plt.legend(loc="lower right")
    plt.show()

def get_report(res_dict, policy_list):
    policy_list = ["balanced", policy_list[-1]]
    train_data, test_data = import_data()
    test_dist = np.array(test_data["distribution"]["test"])
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    train_dist = sum(train_dist)/N
    nsample_train = [train_data["num_samples"][n] for n in range(N)]
    train_support = [int(q * sum(nsample_train)) for q in train_dist]
    bar_colors = [
        "tab:blue",
        "tab:red",
        "tab:olive",
        "tab:green",
        "tab:purple",
        "tab:orange",
        "tab:brown",
        "tab:pink",
        "tab:grey",
        "tab:cyan",
    ]
    values_top = []
    values_bottom = []
    for policy in policy_list:
        res = res_dict[policy]["report"]
        class_list = np.array(list(res.keys())[:-3])
        support_list = np.array([test_dist[int(k)]/train_dist[int(k)] for k in class_list])
        top = 30
        top_index = np.argsort(support_list)[-top:][::-1]
        bottom_index = np.argsort(support_list)[:top]

        major_class = class_list[top_index]
        minor_class = class_list[bottom_index]
        # major_precision = [res[k]["precision"] for k in major_class]
        # major_recall = [res[k]["recall"] for k in major_class]
        major_f1 = [res[k]["precision"] for k in major_class]
        minor_f1 = [res[k]["precision"] for k in minor_class]

        values_top.append(major_f1)
        values_bottom.append(minor_f1)

    
    categories = [f"{support_list[int(k)]:.2f}" for k in major_class]
    categories = [f"{k}" for k in major_class]

    style_list = ["b-", "r-", "y-", "g-", "m-"]
    group_labels = policy_list
    n_categories = len(categories)
    n_groups = len(group_labels)

    index = np.arange(n_categories)
    bar_width = 0.2

    fig, ax = plt.subplots()
    for i in range(n_groups):
        plt.bar(index + i * bar_width, values_top[i], bar_width, label=group_labels[i], color=bar_colors[i])

    ax.set_ylabel('Score')
    ax.set_xticks(index + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(categories)
    ax.set_title(f"top-{top} major classes performance")
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def csprofile(client_cnt, N, m, T):
    limit = (m/N)
    plt.figure()
    plt.axvline(x = limit, color = 'r', linestyle = "--", label = 'participation rate lower bound')
    binwidth = 0.0005
    print(T)
    data = client_cnt
    print(len([x for x in data if x > limit]))
    density, bins, _ = plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    count, _ = np.histogram(data, bins)
    for x,y,num in zip(bins, density, count):
        if num != 0:
            plt.text(x, y+0.05, num, fontsize=10) # x,y,str
    # plt.bar_label(bars)
    plt.xlabel("Participation rate", fontsize=10)
    plt.xticks(np.arange(min(data), max(data) + binwidth*5, binwidth*5), rotation=45)
    plt.ylabel("Client count", fontsize=10)

    plt.show()

def plot_sampling(res_dict, policy_list):
    first_key = policy_list[0]
    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_data, test_data = import_data()
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    top_pick = np.argmax(res_dict["KL_balanced"]["client_cnt"])
    pick_num = int(res_dict["KL_balanced"]["client_cnt"][top_pick] * 2000)
    xtrain = train_data["user_data"][user_name[top_pick]]["x"]
    ytrain = train_data["user_data"][user_name[top_pick]]["y"]
    ytest = test_data["user_data"]["test"]["y"]
    freq = np.zeros(K)
    for i in range(5):
        batch_idx = np.random.choice(len(ytrain), 50, replace=False)
        for b in batch_idx:
            label = ytrain[b]
            freq[label] += 1
    
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
    label_list = policy_list
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    # for outidx, subfig in enumerate(subfigs.flat):
    # subfigs[0].suptitle(f'Time-average label distribution of selected clients')
    ax = subfigs[0].subplots(1, 1)
    # plt.subplots_adjust(top=0.89,
    #                     bottom=0.125,
    #                     left=0.125,
    #                     right=0.9,
    #                     hspace=0.25,
    #                     wspace=0.25)
    # for inidx, ax in enumerate(axs.flat):
    policy = policy_list[-1]
    client_cnt = np.array(res_dict[policy]["client_cnt"])*2000
    policy_dist = np.matmul(client_cnt, train_dist)
    ax.bar(class_name, policy_dist, color=bar_colors, width=0.9)
    ax.title.set_text(f"{label_list[-1]}")
    ylim = 250
    ax.set_ylim([0, ylim])
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")
    
    ax = subfigs[1].subplots(1, 1)
    ax.bar(class_name, test_dist*len(ytest), color=bar_colors, width=0.9)
    ax.title.set_text("Testing dataset")
    ax.set_ylim([0, ylim])
    ax.set_ylabel("Count")
    ax.set_xlabel("Label")

    plt.show()


def plot_allacc():
    alpha_num = [1, 0.5, 0.3]
    alpha_list = ["alpha1", "alpha05", "alpha03"]
    policy_list = ["uniform", "POC", "CBS", "KL"]
    style_list = ["b-", "r-", "y-", "g-"]
    label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    xticks_list = [0, 100, 200, 300, 400, 500]

    T = 500
    linew = 0.8
    # plt.grid()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # fig.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.895,
                        bottom=0.25,
                        left=0.095,
                        right=0.905,
                        hspace=0.2,
                        wspace=0.38)
    # plt.subplot_tool()

    for alpha_idx, alpha_str in enumerate(alpha_list):
        ax[alpha_idx].grid()
        with open(f"results/{dataset}_{alpha_str}_{version}.json", "r") as infile:
            results = json.load(infile) 
            for idx, policy in enumerate(policy_list):
                cur = ax[alpha_idx]
                cur.plot(
                    np.linspace(0, T, T), results[policy]["acc_test"], style_list[idx], linewidth=linew, label=label_list[idx]
                )
                cur.title.set_text(r"$\alpha_{test}$ = " + f"{alpha_num[alpha_idx]}")
                cur.set_ylim([0.2, 0.65])
                cur.set_xticks(xticks_list)
                cur.set_xlabel("Number of rounds")
                cur.set_ylabel("Testing Accuracy")


    fig.legend(label_list, loc='lower center', ncol=4)
    plt.show()    

def plot_allf1():
    alpha_num = [1, 0.5, 0.3]
    alpha_list = ["alpha1", "alpha05", "alpha03"]
    policy_list = ["uniform", "POC", "CBS", "KL"]
    style_list = ["b-", "r-", "y-", "g-"]
    label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    xticks_list = [0, 100, 200, 300, 400, 500]

    T = 500
    linew = 0.8
    # plt.grid()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    # fig.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.895,
                        bottom=0.25,
                        left=0.095,
                        right=0.905,
                        hspace=0.2,
                        wspace=0.38)
    # plt.subplot_tool()

    for alpha_idx, alpha_str in enumerate(alpha_list):
        ax[alpha_idx].grid()
        with open(f"results/{dataset}_{alpha_str}_{version}.json", "r") as infile:
            results = json.load(infile) 
            for idx, policy in enumerate(policy_list):
                cur = ax[alpha_idx]
                cur.plot(
                    np.linspace(0, T, T), results[policy]["f1_test"], style_list[idx], linewidth=linew, label=label_list[idx]
                )
                cur.title.set_text(r"$\alpha_{test}$ = " + f"{alpha_num[alpha_idx]}")
                cur.set_ylim([0.2, 0.5])
                cur.set_xlabel("Number of rounds")
                cur.set_ylabel("Testing macro F1")


    fig.legend(label_list, loc='lower center', ncol=4)
    plt.show() 

def plot_shift(res_dict, policy_list):
    T = len(res_dict[policy_list[0]]["acc_train"])
    
    npolicy = len(policy_list)
    style_list = ["b-", "r-", "y-", "g-", "m-"]
    label_list = policy_list
    idx = 0
    for policy in policy_list:
        shift_test = res_dict[policy]["shift"]
        plt.plot(list(range(T)), shift_test, style_list[idx])
        idx += 1
    plt.legend()
    plt.show()

def get_metric_ovr():
    seed_list = [0, 1, 2]
    Nseed = len(seed_list)
    # subdir = "Tinyimagenet_test"
    # dataset = "Tinyimagenet"
    # alpha = "alpha005"
    # version = "test"

    subdir = "CIFAR100_hypertune"
    dataset = "CIFAR100"
    alpha = "alpha01"
    version = "v1"

    print(f"Plotting results from {dataset} with version:{version} and alpha: {alpha}")

    V_list = [1, 2, 5, 10, 20, 50, 100, 200]
    lambda_list = [0.1, 1, 10, 100]
    # alpha_list = [0.825, 0.875]
    
    policy_list = ["balanced", "POC_balanced", f"KL_balanced_V20_R0125_maxlim"]
    
    policy_list = ["uniform", "POC", "CBS", f"KL_V50_R025_maxlim"]
    policy_list = ["balanced", "POC_balanced", "CBS_balanced", f"KL_balanced_V50_R025_maxlim"]


    # policy_list = [f"KL_balanced_V{v_val}_R1_maxlim" for v_val in V_list]
    policy_list = [f"CBS_balanced_lambda{expfactor}" for expfactor in lambda_list]

    # policy_list = ["uniform", "POC", "CBS", f"KL_V50_R025_maxlim"]
    # policy_list = [f"KL_balanced_V100_R05_diminish16v2"]


    # policy_list = [f"KL_balanced_V50_R025_maxlim", f"KL_balanced_V10_R05_maxlim"]
    for policy in policy_list:
        acc_test_list = {}
        R2R_test_list = {}
        loss_test_list = {}


        target_acc = 0.3244
        
        print(f"Policy {policy}")

        for rseed in seed_list:
            res_dict = {}
            with open(f"results/{subdir}/{dataset}_{alpha}_{version}_S{rseed}_{policy}.json", "r") as infile:
                result = json.load(infile)
                res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
                for i, metric in enumerate(res_dict[policy]):
                    res_dict[policy][metric] = result[metric]
            acc_test_list[rseed] = np.max(res_dict[policy]["acc_test"])
            loss_test_list[rseed] = np.max(res_dict[policy]["loss_test"])
            T = len(res_dict[policy]["acc_test"])
            # if acc_test_list[rseed] < 0.2:
            #     acc_test_list[rseed] = np.mean(acc_test_list[:rseed])

            print(f"seed {rseed}: Accuracy = {acc_test_list[rseed]}")
            R2R_test_list[rseed] = T
            for itr, v in enumerate(res_dict[policy]["acc_test"]):
                    if v >= target_acc:
                        R2R_test_list[rseed] = itr
                        break
    
        print(f"Accuracy Avg: {np.mean(list(acc_test_list.values()))} " 
              f"SD: {np.std(list(acc_test_list.values()))} " 
              f"SE: {np.std(list(acc_test_list.values()))/np.sqrt(Nseed)}"
        )
        # print(f"Accuracy Macro F1: {np.mean(f1_test_list)} SD: {np.std(f1_test_list)}")
        print(f"Accuracy R to reach {target_acc}: {np.mean(list(R2R_test_list.values()))} " 
              f"SD: {np.std(list(R2R_test_list.values()))}"
        )

    


def main():
    N = 5000
    K = 100
    m = 5

    subdir = "CIFAR100_hypertune"
    dataset = "CIFAR100"
    alpha = "alpha01"
    version = "v1"

    KL_name= "KL_balanced_V50_R025_maxlim"
    policy_list = ["balanced", "POC_balanced", "CBS_balanced", KL_name]
    # policy_list = ["uniform", "POC", "CBS", f"KL_V50_R025_maxlim"]

    policy_list = [f"CBS_balanced_lambda0.1"]
    # policy_list = ["balanced"]
    res_dict = {}
    
    for policy in policy_list:
        with open(f"results/{subdir}/{dataset}_{alpha}_{version}_S0_{policy}.json", "r") as infile:
            result = json.load(infile)
            res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
            for i, metric in enumerate(res_dict[policy]):
                res_dict[policy][metric] = result[metric]
    
    T = len(res_dict[policy_list[0]]["acc_test"])
    get_metric_ovr()
    plot_acc(res_dict, policy_list, showtrain=False, Debug=True)
    # plot_f1(res_dict, policy_list)
    # plot_distribution(res_dict, policy_list=[KL_name, KL_name2])
    csprofile(res_dict[policy_list[0]]["client_cnt"], N, m, T)
    # plot_sampling(res_dict, policy_list)
    # plot_shift(res_dict, policy_list)


if __name__ == "__main__":
    main()
