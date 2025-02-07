import json
import numpy as np
import matplotlib.pyplot as plt
from util.misc import L1

N = 10
m = 10


def import_data():
    train_path = "data/FMNIST/FMNIST_N100alpha10_train.json"
    test_path = "data/FMNIST/FMNIST_N100alpha10_test.json"
    # train_path = "data/synthetic/syntha1b1_train.json"
    # test_path = "data/synthetic/syntha1b1_test.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


def plot_distribution(cnt_uni, cnt_QS, cnt_POC, cnt_CBS):
    train_data, test_data = import_data()
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    test_dist = np.array(
        [test_data["distribution"][uname] for uname in test_data["distribution"]]
    )

    Pdist_uni = np.matmul(cnt_uni / m, train_dist)
    Pdist_QS = np.matmul(cnt_QS / m, train_dist)
    Pdist_POC = np.matmul(cnt_POC / m, train_dist)
    Pdist_CBS = np.matmul(cnt_CBS / m, train_dist)

    train_dist_avg = sum(train_dist) / N
    # sumsz = sum([test_data["num_samples"][n] for n in range(N)])
    # szfraction = np.array([test_data["num_samples"][n] / sumsz for n in range(N)])
    # test_dist = np.matmul(np.reshape(szfraction, (1, N)), test_dist).reshape(10)
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

    ylim = max(
        max(test_dist), max(Pdist_uni), max(Pdist_QS), max(Pdist_POC), max(Pdist_CBS)
    )
    for policy in ["uni", "QS", "POC", "CBS"]:
        fig, ax = plt.subplots(1, 2)
        if policy == "uni":
            fig.suptitle(f"Distance of FedAVG = {L1(Pdist_uni, test_dist)}")
            ax[0].bar(class_name, Pdist_uni, color=bar_colors, width=0.9)
        elif policy == "QS":
            fig.suptitle(f"Distance of FedQS = {L1(Pdist_QS, test_dist)}")
            ax[0].bar(class_name, Pdist_QS, color=bar_colors, width=0.9)
        elif policy == "POC":
            fig.suptitle(f"Distance of FedPOC = {L1(Pdist_POC, test_dist)}")
            ax[0].bar(class_name, Pdist_POC, color=bar_colors, width=0.9)
        elif policy == "CBS":
            fig.suptitle(f"Distance of FedCBS = {L1(Pdist_CBS, test_dist)}")
            ax[0].bar(class_name, Pdist_CBS, color=bar_colors, width=0.9)

        ax[0].set_xlabel("Label")
        ax[0].set_ylabel("pmf", rotation=0, ha="right")
        ax[0].set_ylim([0, ylim])

        ax[1].set_xlabel("Label")
        ax[1].set_ylabel("pmf", rotation=0, ha="right")
        ax[1].set_ylim([0, ylim])
        ax[1].bar(class_name, test_dist, color=bar_colors, width=0.9)
        plt.show()


def plot_loss(
    loss_train_iid,
    loss_test_iid,
    loss_train_noniid,
    loss_test_noniid,
    showtrain=False,
):
    T = len(loss_test_iid)
    linew = 0.8
    plt.figure(figsize=(3, 3))
    plt.grid()

    plt.plot(
        np.linspace(0, T, T),
        loss_test_iid,
        "b-",
        linewidth=linew,
        label=r"FedAvg IID",
    )
    plt.plot(
        np.linspace(0, T, T),
        loss_test_noniid,
        "r-",
        linewidth=linew,
        label=r"FedAvg Non-IID",
    )

    # plt.title(r"Global loss of MNIST dataset")
    plt.xlabel("# Communication rounds", fontsize=12)
    plt.ylabel("Global Loss", fontsize=12, ha="right")
    plt.legend(loc="upper right")
    plt.show()


def plot_acc(
    acc_train_iid,
    acc_test_iid,
    acc_train_noniid,
    acc_test_noniid,
    showtrain=False,
):
    T = len(acc_test_iid)
    linew = 0.8
    plt.figure(figsize=(5, 5))
    plt.grid()

    plt.plot(
        np.linspace(0, T, T),
        acc_test_iid,
        "b-",
        linewidth=linew,
        label=r"FedAvg IID",
    )
    plt.plot(
        np.linspace(0, T, T),
        acc_test_noniid,
        "r-",
        linewidth=linew,
        label=r"FedAvg Non-IID",
    )

    # plt.title(r"Test Accuracy of MNIST dataset")
    plt.xlabel("# Communication rounds", fontsize=12)
    plt.ylabel("Test accuracy", fontsize=12)
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


def main():
    with open("results/sim_result/MNIST/N10IID.json", "r") as infile:
        result = json.load(infile)

    loss_train_iid = result["loss_train_uni"][:500]
    loss_test_iid = result["loss_test_uni"][:500]

    acc_train_iid = result["acc_train_uni"][:500]
    acc_test_iid = result["acc_test_uni"][:500]

    with open("results/sim_result/MNIST/N10nonIID.json", "r") as infile:
        result = json.load(infile)

    loss_train_noniid = result["loss_train_uni"]
    loss_test_noniid = result["loss_test_uni"]

    acc_train_noniid = result["acc_train_uni"]
    acc_test_noniid = result["acc_test_uni"]

    plot_loss(
        loss_train_iid,
        loss_test_iid,
        loss_train_noniid,
        loss_test_noniid,
        showtrain=False,
    )

    plot_acc(
        acc_train_iid,
        acc_test_iid,
        acc_train_noniid,
        acc_test_noniid,
        showtrain=False,
    )

    # plot_distribution(cnt_uni, cnt_QS, cnt_POC, cnt_CBS)
    # plot_table(acc_class_uni, acc_class_QS, acc_class_POC, acc_class_CBS)


if __name__ == "__main__":
    main()
