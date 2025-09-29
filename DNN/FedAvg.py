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
import time

import json
import multiprocessing
import pickle
from tqdm import tqdm
from util.cspolicy import (
    client_sample_uni,
    client_sample_POC,
    client_sample_CBS,
    client_sample_KL,
    client_sample_ODFL
)
import models
from util.misc import KL
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def import_data(data_config, rseed):
    dataset, train_group, test_group, v = data_config
    train_path = f"data/Federated/{dataset}_train{train_group}_test{test_group}_{v}_train.pck"
    test_path = f"data/Federated/{dataset}_train{train_group}_test{test_group}_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

def client_selector(policy, metric):
    queue_backlog = metric["queue_backlog"]
    if "uniform" in policy:
        participants_set = client_sample_uni(metric)
    elif "POC" in policy:
        participants_set = client_sample_POC(metric)
    elif "CBS" in policy:
        participants_set = client_sample_CBS(metric, expfactor=10)
    elif "ODFL" in policy:
        participants_set = client_sample_ODFL(metric)
    elif "KL" in policy:
        if "balanced" in policy:
            V = 14.44
        else:
            V = 250
        participants_set, queue_backlog= client_sample_KL(
            metric, V=V, min_rate_lim=True, max_rate_lim=True
        )
    return participants_set, queue_backlog

def client_train(client_idx, model, x_train, y_train, stepsize, max_steps, B, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    history = model.fit(
        x_train,
        y_train,
        batch_size=B,
        epochs=max_steps,
        steps_per_epoch = 1,
        shuffle=True,
        verbose=0,
        class_weight = weight
    )
    return (client_idx, history, model)

def model_aggregation(weights):
    agg_weights = list()
    for weights_list_tuple in zip(*weights): 
        agg_weights.append(
            np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        )
    return agg_weights

def FedAvg(env_config, data_config, policy, rseed, Debug=True, early_stopping=True):
    
    N, r, m, B, K, I, T = env_config
    print(f"Algorithm: {policy} Random Seed: {rseed} Early Stopping: {early_stopping}")
    np.random.seed(rseed)

    set_gpu_memory_growth()

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        initializer=set_gpu_memory_growth,  
        mp_context=multiprocessing.get_context("spawn") 
    )

    log_dir = "logs/fl_global_model"
    summary_writer = tf.summary.create_file_writer(log_dir)

    policy_name = policy
    
    best_accuracy = 0.00
    wait = 0
    patience = 100
    min_delta = 0.0005

    decay_rate = 0.9992
    stepsize = 0.01

    Aq = np.zeros(N)
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

    timer_dict = {
        "Client_selection": list(),
        "Data_loading": list(),
        "Local_training": list(),
        "Model_aggregation": list(),
        "Model_evaluation": list()
    }

    train_data, test_data = import_data(data_config, rseed)

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    nsample_train = [train_data["num_samples"][n] for n in range(N)]
    max_steps = int(max(nsample_train)/B) * I
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sum(nsample_train) for n in range(N)]
    )
    
    test_dist = np.array(test_data["distribution"]["test"])

    client_weight = np.array([0.5/N for n in range(N)])

    
    # x_train = [tf.convert_to_tensor(train_data["user_data"][user_name[n]]["x"]) for n in range(N)]
    # y_train = [tf.convert_to_tensor(train_data["user_data"][user_name[n]]["y"]) for n in range(N)]
    x_test = tf.convert_to_tensor(test_data["user_data"]["test"]["x"])
    y_test = tf.convert_to_tensor(test_data["user_data"]["test"]["y"])

    test_batch_sz = B
    
    max_w = 0
    wdict = [
        {
            k: (
                test_dist[k] / train_dist[n][k]
                if train_dist[n][k] > 0
                else max_w
            )
            for k in range(K)
        }
        for n in range(N)
    ]

    if policy == "Fedprox":
        global_model = models.get_LeNet(config="prox")
    else:
        global_model = models.get_LeNet()
        local_model = models.get_LeNet()
    
    for t in tqdm(range(T)):
        keras.backend.clear_session()

        start = time.perf_counter()

        available_client = np.random.choice(N, r, replace=False)
        train_metric = {
            "round": t+1,
            "train_dist":train_dist,
            "test_dist":test_dist,
            "available_client": available_client,
            "n_participants": m,
            "client_sample": nsample_train,
            "client_weight": client_weight,
            "client_count": client_cnt + 1,
            "client_loss": client_loss_train,
            "queue_backlog": (Aq, Yq, Zq)
        }

        participants_set, queue_backlog = client_selector(policy, train_metric)    
        Aq, Yq, Zq = queue_backlog

        duration = time.perf_counter() - start
        timer_dict["Client_selection"].append(duration)

        weights = []
        futures = []

        start = time.perf_counter()
        for n in participants_set:
            client_cnt[n] += 1

            local_model.set_weights(global_model.get_weights())

            if "balanced" in policy:
                class_weight = wdict[n]
            else: 
                class_weight = None

            x_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["x"])
            y_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["y"])
            futures.append(
                executor.submit(
                    client_train, n, local_model, x_train, y_train, stepsize, max_steps, B, class_weight
                )
            )

        
        for future in concurrent.futures.as_completed(futures):
            client_idx, history, local_model = future.result()
            client_loss_train[client_idx] = history.history["loss"][-1]
            client_acc_train[client_idx] = history.history["sparse_categorical_accuracy"][-1]
            weights.append(local_model.get_weights())

        duration = time.perf_counter() - start
        timer_dict["Local_training"].append(duration)

        start = time.perf_counter()

        new_weights = model_aggregation(weights)            
        global_model.set_weights(new_weights)

        duration = time.perf_counter() - start
        timer_dict["Model_aggregation"].append(duration)

        del weights, new_weights
        # del x_train, y_train
        gc.collect()


        start = time.perf_counter()
        test_scores = global_model.evaluate(x_test, y_test, verbose=0, batch_size=test_batch_sz)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]
        
        model_pred = global_model.predict(x_test, verbose=0)
        y_pred = tf.convert_to_tensor([np.argmax(v) for v in model_pred])

        f1_test[t] = f1_score(y_test, y_pred, average="macro")
        
        policy_dist = np.matmul(client_cnt/((t+1)*m), train_dist)
        shift_test[t] = KL(test_dist, policy_dist)

        masked = np.isfinite(client_loss_train)
        loss_train[t] = sum(client_loss_train[masked])/max(sum(masked), 1)
        acc_train[t] = np.dot(client_acc_train, szfrac_train)

        duration = time.perf_counter() - start
        timer_dict["Model_evaluation"].append(duration)
        # Write scalar summaries
        with summary_writer.as_default():
            tf.summary.scalar("Accuracy", float(acc_test[t]), step=t)
            tf.summary.scalar("Loss", float(loss_test[t]), step=t)
            summary_writer.flush()

        if Debug:
            print(f'{policy_name}: train loss:{loss_train[t]:.4f}, train acc: {acc_train[t]:.4f}, test loss: {loss_test[t]:.4f}, test acc: {acc_test[t]:.4f}, test macro f1: {f1_test[t]:.4f}, sumrate: {sum(client_cnt)}')
        
        train_log = {
            'epoch': t,
            'loss_train': list(loss_train),
            'loss_test':list(loss_test),
            'acc_train': list(acc_train),
            'acc_test': list(acc_test),        
            'client_cnt': list(client_cnt),
            'f1_test': list(f1_test),
            'shift': list(shift_test),
            'Aq' : list(Aq),
            'Yq' : list(Yq),
            'Zq' : list(Zq),
            'client_loss_train' : list(client_loss_train),
            'client_acc_train' : list(client_acc_train),
            'best_accuracy' : best_accuracy,
            'stepsize': stepsize,
            'policy_name': policy_name
        }
            
        if early_stopping:
            if acc_test[t] - best_accuracy > min_delta:
                best_accuracy = acc_test[t]
                wait = 0  
            else:
                wait += 1  

            if wait >= patience:
                print(f"\nEarly stopping triggered after {t + 1} rounds!")
                break  
        
                
        stepsize = stepsize * decay_rate
        bkd.clear_session()

    print([ (key, np.mean(timer_dict[key])) for key in timer_dict.keys()])
    del global_model, y_pred, model_pred
    gc.collect()
    bkd.clear_session()
    return train_log
