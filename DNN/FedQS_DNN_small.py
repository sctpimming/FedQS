import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_lazy_compilation=false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

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
    client_sample_comb
)
import models
from util.misc import KL
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_client_dataset(uname):
    dataset_dir = f"data/Tinyimagenet_alpha05_control/{uname}/"
    batch_size = B
    if uname == "test":
        batch_size = 256

    dataset = tf.data.Dataset.load(dataset_dir, compression="GZIP")
    dataset_size = sum(1 for _ in dataset)

    dataset = dataset.shuffle(buffer_size=dataset_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_metadata():
    dataset_dir = f"data/Tinyimagenet_alpha05_control/"
    with open(dataset_dir+"train_data.pck", "rb") as f:
        train_data = pickle.load(f)
    with open(dataset_dir+"test_data.pck", "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data


def init(v):
    gpus = tf.config.list_physical_devices('GPU')
    print(len(gpus))
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

def client_train(model, uname, num_samples, stepsize, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    client_dataset = get_client_dataset(uname)
    history = model.fit(
        client_dataset,
        epochs=I,
        # steps_per_epoch = 1,
        verbose=0,
        class_weight = weight,
    )

    del client_dataset
    gc.collect()

    return history, model

N = 10
r = 30
m = 10
B = 64
I = 10
K = 200
T = 25

def FedAvg(policy, rseed, Debug=True):
    print(f"Algorithm: {policy} Random Seed: {rseed}")
    np.random.seed(rseed)

    Aq = np.zeros(N)
    Bq = 0
    Yq = np.zeros(N)
    Zq = np.zeros(N)

    
    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    f1_test = np.zeros(T)
    shift_test = np.zeros(T)

    client_cnt = np.zeros(N)
    client_loss_train = np.ones(N) * np.inf
    client_acc_train = np.zeros(N)

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_data, test_data = get_metadata()
    train_dist = np.array(
        [train_data[uname]["distribution"] for uname in user_name]
    )
    nsample_train = np.array(
        [train_data[uname]["num_samples"] for uname in user_name]
    )
    sumsz_train = sum(nsample_train)
    szfrac_train = nsample_train/sumsz_train

    test_dataset = get_client_dataset(uname="test")
    test_dist = np.array(test_data["test"]["distribution"])
    Q_inv = np.array([1/q if q > 0 else 10 ** (-20) for q in test_dist])

    decay_rate = 1
    stepsize = 0.01

    if "Fedprox" in policy:
        global_model = models.get_LeNet(config="prox")
        local_model = models.get_LeNet(config="prox")
    else:
        global_model = models.get_ResNet50(weight_config=None, K=200)
        local_model = models.get_ResNet50(weight_config=None, K=200)
    
    for t in tqdm(range(T)):
        bkd.clear_session()

        global_weights = [np.zeros_like(w) for w in global_model.get_weights()]

        masked = np.isfinite(client_loss_train)
        loss_train[t] = sum(client_loss_train[masked])/max(sum(masked), 1)
        acc_train[t] = np.dot(client_acc_train, szfrac_train)

        if policy == "POC" or policy == "POC_balanced":
            available_client = np.random.choice(N, N, replace=False, p=szfrac_train)
        else:
            available_client = list(range(N)) 
        
        if policy == "uniform" or policy == "balanced" or "Fedprox" in policy:
            participants_set = client_sample_uni(available_client, m)
        elif policy == "CBS" or policy == "CBS_balanced":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, nsample_train, t, client_cnt + 1
            )
        elif policy == "KL" or policy == "KL_balanced":
            participants_set, Aq, Yq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, nsample_train,
                Aq, Yq, Zq, Q_inv, 
                V=10, R=(m/N)*0.5, 
                max_rate_lim=True, diminish=False
            )
        elif policy == "POC" or policy == "POC_balanced":
            participants_set = client_sample_POC(available_client, m, client_loss_train)

        for n in participants_set:

            bkd.clear_session()

            client_cnt[n] += 1

            local_model.set_weights(global_model.get_weights())

            if "balanced" in policy:
                max_w = 0.0
                wdict = {
                    k: (
                        test_dist[k] / train_dist[n][k]
                        if train_dist[n][k] > 0
                        else max_w
                    )
                    for k in range(K)
                }
                history, local_model = client_train(
                    local_model, user_name[n], nsample_train[n], stepsize, weight=wdict
                )
            else:
                history, local_model = client_train(local_model, user_name[n], nsample_train[n], stepsize)
            client_loss_train[n] = history.history["loss"][-1]
            client_acc_train[n] = history.history["sparse_categorical_accuracy"][-1]
            print(f"{user_name[n]}  loss: {client_loss_train[n]} acc: {client_acc_train[n]}")
            client_weights = local_model.get_weights()
            global_weights = [g + (c / m) for g, c in zip(global_weights, client_weights)]

            del client_weights, history
            gc.collect()
            
        global_model.set_weights(global_weights)

        test_scores = global_model.evaluate(test_dataset, verbose=0)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]
                
        policy_dist = np.matmul(client_cnt/((t+1)*m), train_dist)
        shift_test[t] = KL(test_dist, policy_dist)

        if Debug:
            print(f'{policy}: train loss:{loss_train[t]:.4f}, train acc: {acc_train[t]:.4f}, test loss: {loss_test[t]:.4f}, test acc: {acc_test[t]:.4f}')

        stepsize = stepsize * decay_rate

        del global_weights
        gc.collect()
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
        list(shift_test),
        policy
    )


def main():

    seed_list = [1, 2, 3, 4]
    run_list = ["control"]
    for v in run_list:
        print(f"Running version {v} with the following seed {seed_list}")
        policy_list = ["uniform"]
        for rseed in seed_list:
            with concurrent.futures.ProcessPoolExecutor(initializer=init, initargs=(v, ), max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = [executor.submit(FedAvg, policy, rseed) for policy in policy_list]
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    if "KL" in results[-1]:
                        policy_name = f"{results[-1]}_V10_R05_maxlim"
                    else:
                        policy_name = f"{results[-1]}"
                    results = results[:-1]
                    print(f"\n {policy_name} completed")
                    res_dict= {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[], "shift":[]}
                    for i, metric in enumerate(res_dict):
                        res_dict[metric] = results[i]
                    with open(f"results/Tinyimagenet_alpha05_{v}_S{rseed}_{policy_name}.json", "w") as f:
                        json.dump(res_dict, f)  


if __name__ == "__main__":
    main()
