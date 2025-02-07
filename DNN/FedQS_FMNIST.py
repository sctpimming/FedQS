import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
sys.path.append('code')
from numba import cuda
cuda.select_device(0)
cuda.close()

import numpy as np
import tensorflow as tf
import keras
from keras import backend as bkd
from sklearn.metrics import f1_score

import json
import multiprocessing
import pickle
from tqdm import tqdm
from util.cspolicy import (
    client_sample_uni,
    client_sample_POC,
    client_sample_CBS,
    client_sample_QS,
    client_sample_KL,
)
from models import get_MLPmodel, get_CNNmodel, get_CNNmodel_cho

def import_data(v):
    train_path = f"data/Federated/FMNIST_alpha05_{v}_train.pck"
    test_path = f"data/Federated/FMNIST_alpha05_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def init(v):
    global train_data
    global test_data
    train_data, test_data = import_data(v)


def client_train(model, x_train, y_train, stepsize):
    bkd.set_value(model.optimizer.learning_rate, stepsize)
    model.fit(
        x_train,
        y_train,
        batch_size=B,
        epochs=I,
        steps_per_epoch=1,
        shuffle=True,
        verbose=0,
    )
    return model


np.random.seed(12345)

N = 100
r = 30
m = 15
B = 50
I = 5
K = 10
T = 500
w_num = 10


def FedAvg(policy, eps=np.log(2)):
    global_model = get_CNNmodel(img_shape=(28, 28, 1))
    Aq = np.zeros(N)
    Zq = np.zeros(N)

    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    f1_test = np.zeros(T)
    client_cnt = np.zeros(N)


    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    Q = np.ones((K, K))
    for k in range(K):
        Q[k][k] = np.exp(eps)
    Q = Q / (K - 1 + np.exp(eps))
    M = np.zeros((N, K))
    for n in range(N):
        uname = "f_{0:05d}".format(n)
        y_list = train_data["user_data"][uname]["y"]
        for y in y_list:
            output = np.random.choice(10, p=Q[y][:])
            M[n][output] += 1
        M[n][:] = M[n][:] / len(y_list)
    train_dist = ((np.exp(eps) + K - 1) / (np.exp(eps) - 1) * M) - (1 / (np.exp(eps) - 1))
    # train_dist = np.array(
    #     [train_data["distribution"][uname] for uname in train_data["distribution"]]
    # )
    sumsz_train = sum([train_data["num_samples"][n] for n in range(N)])
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )
    test_dist = np.array(test_data["distribution"]["test"])

    x_test = tf.convert_to_tensor(test_data["user_data"]["test"]["x"])
    y_test = tf.convert_to_tensor(test_data["user_data"]["test"]["y"])
    
    shape_list = []
    for layer in global_model.layers:
        for w in layer.get_weights():
            w_shape = np.shape(w)
            shape_list.append(np.zeros(w_shape))
    w_num = len(shape_list)
    decay_rate = 0.9992
    stepsize = 0.01
    for t in tqdm(range(T)):
        w_agg = dict(
            zip(
                list(range(w_num)),
                [
                    np.zeros((5, 5, 1, 6)),
                    np.zeros((6,)),
                    np.zeros((5, 5, 6, 16)),
                    np.zeros((16,)),
                    np.zeros((1024, 120)),
                    np.zeros((120,)),
                    np.zeros((120, 84)),
                    np.zeros((84,)),
                    np.zeros((84, 10)),
                    np.zeros((10,)),
                ],
            )
        )
        client_eval = [
            global_model.evaluate(
                np.array(train_data["user_data"][user_name[n]]["x"]),
                np.array(train_data["user_data"][user_name[n]]["y"]),
                verbose=0,
                batch_size=B
            )
            for n in range(N)
        ]
        client_loss_train = [v[0] for v in client_eval]
        client_acc_train = [v[1] for v in client_eval]
        loss_train[t] = sum(client_loss_train) / N
        acc_train[t] = sum(client_acc_train) / N

        if policy == "POC":
            available_client = np.random.choice(N, r, replace=False, p=szfrac_train)
        else:
            available_client = list(range(N)) 

        if policy == "uniform":
            participants_set = client_sample_uni(available_client, m)
        elif policy == "CBS":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, B, t, client_cnt + 1
            )
        elif policy == "QS":
            participants_set, Queue = client_sample_QS(
                train_dist, test_dist, available_client, m, Queue
            )
        elif policy == "KL":
            participants_set, Aq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, Aq, Zq
            )
        elif policy == "POC":
            participants_set = client_sample_POC(available_client, m, client_loss_train)

        for n in participants_set:
            client_cnt[n] += 1

            x_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["x"])
            y_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["y"])

            local_model = client_train(global_model, x_train, y_train, stepsize)
            for idx in range(w_num):
                w_agg[idx] += local_model.get_weights()[idx]

        for idx in range(w_num):
            w_agg[idx] = w_agg[idx] / m

        idx = 0
        for layer in global_model.layers:
            if layer.name.startswith("flatten") or layer.name.startswith("max") or layer.name.startswith("dropout"):
                continue
            global_model.get_layer(layer.name).set_weights([w_agg[idx], w_agg[idx + 1]])
            idx += 2

        test_scores = global_model.evaluate(x_test, y_test, verbose=0)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]

        model_pred = global_model.predict(x_test, verbose=0)
        y_pred = [np.argmax(v) for v in model_pred]
        f1_test[t] = f1_score(y_test, y_pred, average="macro")

        stepsize = stepsize * decay_rate
        bkd.clear_session()

    return (
        list(loss_train),
        list(loss_test),
        list(acc_train),
        list(acc_test),
        list(client_cnt / T),
        list(f1_test)
    )


def main():
    run_list = ["v1", "v2", "v3"]
    for v in run_list:
        print(f"Running version {v}")
        with multiprocessing.Pool(initializer=init, initargs=(v, ), processes=12) as pool:        
            policy_list = ["uniform", "POC", "CBS", "KL"]
            results = pool.map(FedAvg, policy_list)
            res_dict = {}
            for idx, policy in enumerate(policy_list):
                res_dict[policy] = {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
                for i, metric in enumerate(res_dict[policy]):
                    res_dict[policy][metric] = results[idx][i]
            with open(f"results/FMNIST_alpha05_{v}_epslg2.json", "w") as f:
                json.dump(res_dict, f)


if __name__ == "__main__":
    main()
