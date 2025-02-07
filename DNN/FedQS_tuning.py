import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
sys.path.append('code')
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf
import keras
from keras import backend as bkd
from sklearn.metrics import f1_score, classification_report
import concurrent.futures
import gc


import json
import multiprocessing
import pickle
from tqdm import tqdm
from util.cspolicy import (
    client_sample_uni,
    client_sample_POC,
    client_sample_CBS,
    client_sample_KL,
)
import models
from util.misc import KL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def import_data(v):
    train_path = f"data/Federated/CIFAR100_alpha01_{v}_train.pck"
    test_path = f"data/Federated/CIFAR100_alpha01_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def init(v):   
    global train_data
    global test_data
    train_data, test_data = import_data(v)

def client_train(model, x_train, y_train, stepsize, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    history = model.fit(
        x_train,
        y_train,
        batch_size=B,
        epochs=I,
        steps_per_epoch=1,
        shuffle=True,
        verbose=0,
        class_weight = weight,
    )
    return history, model

seed_list = list(range(5))

N = 5000
r = 3000
m = 5
B = 50
I = 10
K = 100
T = 2500



def FedAvg(policy, param, rseed, Debug=True):
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Algorithm: {policy} param val:{param} Random Seed: {rseed}")
    np.random.seed(rseed)
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)])
    Aq = np.zeros(N)
    Yq = np.zeros(N)
    Zq = np.zeros(N)

    
    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    f1_test = np.zeros(T)
    shift_test = np.zeros(T)
    gradient_norm = np.zeros(T)

    client_cnt = np.zeros(N)
    client_loss_train = np.ones(N) * np.inf
    client_acc_train = np.zeros(N)
    client_gradient_norm = np.zeros(N)

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    nsample_train = [train_data["num_samples"][n] for n in range(N)]
    sumsz_train = sum(nsample_train)
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )
    # test_dist = np.array([1/K]*K)
    test_dist = np.array(test_data["distribution"]["test"])
    Q_inv = np.array([1/q if q > 0 else 10 ** (-20) for q in test_dist])
    # wdict =  {k:test_dist[k] for k in range(K)}

    x_test = tf.convert_to_tensor(test_data["user_data"]["test"]["x"])
    y_test = tf.convert_to_tensor(test_data["user_data"]["test"]["y"])
    test_batch_sz = y_test.shape[0]

    decay_rate = 0.9992
    stepsize = 0.01

    if policy == "Fedprox":
        global_model = models.get_LeNet(config="prox")
    else:
        global_model = models.get_LeNet()
        local_model = models.get_LeNet()
    
    for t in tqdm(range(T)):
        keras.backend.clear_session()
        masked = np.isfinite(client_loss_train)
        loss_train[t] = sum(client_loss_train[masked])/max(sum(masked), 1)
        acc_train[t] = np.dot(client_acc_train, szfrac_train)

        if policy == "POC" or policy == "POC_balanced":
            available_client = np.random.choice(N, r, replace=False, p=szfrac_train)
        else:
            available_client = list(range(N)) 
        
        # print("Sampling available client")

        if policy == "uniform" or policy == "balanced":
            participants_set = client_sample_uni(available_client, m)
        elif policy == "CBS" or policy == "CBS_balanced":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, nsample_train, t, client_cnt + 1, alpha=param
            )
        elif policy == "KL" or policy == "KL_balanced":
            participants_set, Aq, Yq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, nsample_train,
                Aq, Yq, Zq, Q_inv, V=1, R=(m/N)*0.75, max_rate_lim=True
            )
        elif policy == "POC" or policy == "POC_balanced":
            participants_set = client_sample_POC(available_client, m, client_loss_train)




        weights = []
        for n in participants_set:
            # print(n)
            keras.backend.clear_session()
            client_cnt[n] += 1

            local_model.set_weights(global_model.get_weights())

            x_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["x"])
            y_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["y"])
            if "balanced" in policy:
                    max_w = 1000.0
                    wdict = {
                        k: (
                            test_dist[k] / train_dist[n][k]
                            if train_dist[n][k] > 0
                            else max_w
                        )
                        for k in range(K)
                    }
                    history, local_model = client_train(
                       local_model, x_train, y_train, stepsize, weight=wdict
                    )
            else:
                history, local_model = client_train(local_model, x_train, y_train, stepsize)
            # score = local_model.evaluate(x_train, y_train, verbose=0, batch_size=32)

            client_loss_train[n] = history.history["loss"][-1]
            client_acc_train[n] = history.history["sparse_categorical_accuracy"][-1]
            
            # client_gradient_norm[n] = score[2]
            weights.append(local_model.get_weights())


        new_weights = list()

        for weights_list_tuple in zip(*weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
            
        global_model.set_weights(new_weights)
        del weights, new_weights
        del x_train, y_train
        gc.collect()

    
        test_scores = global_model.evaluate(x_test, y_test, verbose=0, batch_size=test_batch_sz)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]
        gradient_norm[t] = np.mean(client_gradient_norm)
        
        model_pred = global_model.predict(x_test, verbose=0)
        y_pred = [np.argmax(v) for v in model_pred]

        f1_test[t] = f1_score(y_test, y_pred, average="macro")
        
        policy_dist = np.matmul(client_cnt/((t+1)*m), train_dist)
        shift_test[t] = KL(test_dist, policy_dist)
        # if acc_test[t] >= max(acc_test):
        #     report = classification_report(y_test, y_pred, zero_division=0.0, output_dict=True)

        if Debug:
            print(f'{policy}: train loss:{loss_train[t]:.4f}, train acc: {acc_train[t]:.4f}, test loss: {loss_test[t]:.4f}, test acc: {acc_test[t]:.4f}, test macro f1: {f1_test[t]:.4f}, gradient norm: {gradient_norm[t]:.4f}')


        stepsize = stepsize * decay_rate
        bkd.clear_session()

    del global_model
    gc.collect()
    bkd.clear_session()

    return (
        list(loss_train),
        list(loss_test),
        list(acc_train),
        list(acc_test),
        list(client_cnt / T),
        list(f1_test),
        list(gradient_norm),
        list(shift_test),
        param
    )


def main():
    Nseed = 5
    seed_list = [3, 4, 5]
    run_list = [[1, 2], [5, 10], [20, 50], [100, 200]]
    run_list = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9]]
    for V_list in run_list:
        v = "v1"
        print(f"Running version {v} with V = {V_list} and seed = {seed_list}")
        # init(v)
        
        for rseed in seed_list:
            with concurrent.futures.ProcessPoolExecutor(initializer=init, initargs=(v, ), max_workers=2, mp_context=multiprocessing.get_context("fork")) as executor:
                futures = [executor.submit(FedAvg, "CBS_balanced", V, rseed) for V in V_list]
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    V = results[-1]
                    policy_name = f"CBS_balanced_alpha{V}"
                    results = results[:-1]
                    print(f"\n {policy_name} completed")
                    res_dict= {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[], "gradient_norm":[], "shift":[]}
                    for i, metric in enumerate(res_dict):
                        res_dict[metric] = results[i]
                    with open(f"results/CIFAR100_hypertune/CIFAR100_alpha01_{v}_S{rseed}_{policy_name}.json", "w") as f:
                        json.dump(res_dict, f)  


if __name__ == "__main__":
    main()
