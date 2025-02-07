import json
import numpy as np
import matplotlib.pyplot as plt
from util.misc import KL
import pickle

N = 100
m = 15

dataset = "FMNIST"
alpha = "alpha1"
version = "v5"

def import_data():
    train_path = f"data/Federated/{dataset}_{alpha}_{version}_train.pck"
    test_path = f"data/Federated/{dataset}_{alpha}_{version}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def plot_distribution(results, Debug=False):
    train_data, test_data = import_data()
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    class_name = [f"{n}" for n in range(10)]
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

    policy_list = ["KL", "QSH", "QS", "QSPlus"]
    # policy_list = ["QSPlus"]
    label_list = ["Function of time avg", "Time avg of function", "Top-m distance (L2)", "Top-m distance (KL)"]
    ylim = max(test_dist)
    fig = plt.figure()
    subfigs = fig.subfigures(1, 2)
    # for outidx, subfig in enumerate(subfigs.flat):
    subfigs[0].suptitle(f'Time average participant distribution')
    axs = subfigs[0].subplots(2, 2)
    for inidx, ax in enumerate(axs.flat):
        policy = policy_list[inidx]
        client_cnt = np.array(results[policy]["client_cnt"])/m
        policy_dist = np.matmul(client_cnt, train_dist)
        print(f"distance from testing distibution using {policy} is {KL(test_dist, policy_dist)}")
        ax.bar(class_name, policy_dist, color=bar_colors, width=0.9)
        ax.title.set_text(f"{label_list[inidx]}")
        ax.set_ylim([0, ylim])
    
    ax = subfigs[1].subplots(1, 1)
    ax.bar(class_name, test_dist, color=bar_colors, width=0.9)
    ax.title.set_text("Testing distribution")
    ax.set_ylim([0, ylim])

    plt.show()
    # for idx, policy in enumerate(policy_list):
    #     fig, ax = plt.subplots(1, 2)
    #     client_cnt = np.array(results[policy]["client_cnt"])/m
    #     policy_dist = np.matmul(client_cnt, train_dist)  
    #     klval = KL(test_dist, policy_dist)
    #     title_str = f"Distance of {label_list[idx]} = {klval}"
    #     fig.suptitle(title_str)
    #     ax[0].bar(class_name, policy_dist, color=bar_colors, width=0.9)

    #     ax[0].set_xlabel("Label")
    #     ax[0].set_ylabel("pmf", rotation=0, ha="right")
    #     ax[0].set_ylim([0, ylim])

    #     ax[1].set_xlabel("Label")
    #     ax[1].set_ylabel("pmf", rotation=0, ha="right")
    #     ax[1].set_ylim([0, ylim])
    #     ax[1].bar(class_name, test_dist, color=bar_colors, width=0.9)
    #     plt.show()
        
    #     if Debug == True:
    #         print(title_str)


def plot_loss(
    results,
    showtrain=False,
):

    T = len(results["QSPlus"]["loss_train"])
    policy_list = ["KL", "QSH", "QS", "QSPlus"]
    # policy_list = ["QSPlus"]

    # policy_list = ["uniform"]
    style_list = ["b-", "r-", "y-", "g-"]
    label_list = ["Function of time avg", "Time avg of function", "Top-m distance (L2)", "Top-m distance (KL)"]
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
    showtrain=False,
    Debug=False
):
    T = len(results["QSPlus"]["acc_train"])
    policy_list = ["KL", "QSH", "QS", "QSPlus"]
    # policy_list = ["QSPlus"]   
    label_list = ["Function of time avg", "Time avg of function", "Top-m distance (L2)", "Top-m distance (KL)"]
    style_list = ["b-", "r-", "y-", "g-"]
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
        target_acc = 0.85
        for idx, policy in enumerate(policy_list):
            print(f"{label_list[idx]} best test accuracy: {max(results[policy]['acc_test'])}")
            for itr, v in enumerate(results[policy]["acc_test"]):
                if v >= target_acc:
                    print(f"{label_list[idx]} takes {itr} iterations to reach {target_acc} accuracy")
                    break

    # plt.title(r"Test Accuracy of MNIST dataset")
    plt.xlabel("Communication rounds", fontsize=10)
    plt.ylabel("Test accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.show()

def plot_f1(
    results,
):
    T = len(results["QSPlus"]["acc_train"])
    policy_list = ["KL", "QSH", "QS", "QSPlus"]
    # policy_list = ["QSPlus"]

    label_list = ["Function of time avg", "Time avg of function", "Top-m distance (L2)", "Top-k distance (KL)"]

    # policy_list = ["uniform", "POC", "CBS", "KL"]
    # policy_list = ["uniform"]
    style_list = ["b-", "r-", "y-", "g-"]
    # label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
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

def plot_table(acc_class_uni, acc_class_QS, acc_class_POC, acc_class_CBS):
    fig, ax = plt.subplots()
    ax.set_axis_off()

    collable = ["class " + str(k) for k in range(10)]
    rowlabel = ["FedAvg", "FedQS", "FedPOC", "FedCBS"]
    acc_stacks = np.zeros((4, 10))
    acc_stacks[0, :] = np.round(acc_class_uni, 3)
    acc_stacks[1, :] = np.round(acc_class_QS, 3)
    acc_stacks[2, :] = np.round(acc_class_POC, 3)
    acc_stacks[3, :] = np.round(acc_class_CBS, 3)

    ax.table(cellText=acc_stacks, colLabels=collable, rowLabels=rowlabel, loc="center")
    fig.tight_layout()
    plt.show()


def csprofile(cnt_uni, cnt_POC, cnt_CBS, cnt_QS):
    plt.figure()
    binwidth = 0.001
    data = cnt_QS
    print(len([x for x in data if x <= 0.01]))
    plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
    plt.xlabel("Participation (%)", fontsize=10)
    plt.ylabel("Client count", fontsize=10)

    plt.show()

def plot_allacc():
    alpha_num = [1, 0.5, 0.3]
    alpha_list = ["alpha1", "alpha05", "alpha03"]
    policy_list = ["uniform", "POC", "CBS", "KL"]
    style_list = ["b-", "r-", "y-", "g-"]
    label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    T = 500
    linew = 0.8
    # plt.grid()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5))
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
                ax[alpha_idx].plot(
                    np.linspace(0, T, T), results[policy]["acc_test"], style_list[idx], linewidth=linew, label=label_list[idx]
                )
                ax[alpha_idx].title.set_text(fr"$\alpha$ = {alpha_num[alpha_idx]}")
                ax[alpha_idx].set_ylim([0.2, 0.7])
                ax[alpha_idx].set_xlabel("Communication rounds")
                ax[alpha_idx].set_ylabel("Accuracy score")


    fig.legend(label_list, loc='lower center', ncol=4)
    plt.show()    

def plot_allf1():
    alpha_num = [1, 0.5, 0.3]
    alpha_list = ["alpha1", "alpha05", "alpha03"]
    policy_list = ["KL", "QSH", "QS", "QSPlus"]
    style_list = ["b-", "r-", "y-", "g-"]
    label_list = ["FedAvg", "Power-of-choice", "Fed-CBS", "FedQS (ours)"]
    T = 500
    linew = 0.8
    # plt.grid()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5))
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
                ax[alpha_idx].plot(
                    np.linspace(0, T, T), results[policy]["f1_test"], style_list[idx], linewidth=linew, label=label_list[idx]
                )
                ax[alpha_idx].title.set_text(fr"$\alpha$ = {alpha_num[alpha_idx]}")
                ax[alpha_idx].set_ylim([0.2, 0.5])
                ax[alpha_idx].set_xlabel("Communication rounds")
                ax[alpha_idx].set_ylabel("F1 score")


    fig.legend(label_list, loc='lower center', ncol=4)
    plt.show() 

def main():
    print(f"Visualize results from dataset {dataset} with alpha = {alpha} from version {version}")
    with open(f"results/{dataset}_{alpha}_{version}_reverse.json", "r") as infile:
        result = json.load(infile)

    plot_loss(result, showtrain=False)
    plot_acc(result, showtrain=False, Debug=True)
    # plot_allf1()
    plot_f1(result)
    plot_distribution(result, Debug=True)
    # plot_table(acc_class_uni, acc_class_QS, acc_class_POC, acc_class_CBS)
    #csprofile(cnt_uni, cnt_POC, cnt_CBS, cnt_QS)


if __name__ == "__main__":
    main()
