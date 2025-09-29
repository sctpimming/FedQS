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


N = 5000
r = int(N * 0.3)
m = 5
B = 50
I = 5
K = 100
T = 1000

dataset = "CIFAR100"
train_group = "G3"
test_group = "G4"
v = "val3"

if dataset == "CIFAR":
    K = 10
elif dataset == "CIFAR100":
    K = 100
elif dataset == "Tinyimagenet":
    K = 200

seed_list = [0]
run_list = np.linspace(10, 50, 9, endpoint=False)
run_list = run_list[1:]
print(run_list)
nworkers = 2
run_list = np.array(run_list).reshape(-1, nworkers)/1
policy = "KL_balanced"


def import_data(v, seed):
    train_path = f"data/Federated/{dataset}_train{train_group}_test{test_group}_{v}_train.pck"
    test_path = f"data/Federated/{dataset}_train{train_group}_test{test_group}_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def init(v, seed):   
    global train_data
    global test_data
    train_data, test_data = import_data(v, seed)

def client_train(model, x_train, y_train, stepsize, max_steps, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    history = model.fit(
        x_train,
        y_train,
        batch_size=B,
        epochs=max_steps,
        steps_per_epoch=1,
        shuffle=True,
        verbose=0,
        class_weight = weight,
    )
    return history, model

def FedAvg(policy, param, rseed, Debug=True, early_stopping=True):
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Algorithm: {policy} param val:{param} Random Seed: {rseed} Early Stopping: {early_stopping}")
    np.random.seed(rseed)
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
    if "KL" in policy:
        policy_name = f"{policy}_V{param}"
    elif "CBS" in policy:
        policy_name = f"{policy}_lambda{param}"
    else:
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

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    nsample_train = [train_data["num_samples"][n] for n in range(N)]
    max_steps = int(max(nsample_train)/B) * I
    print(f"Training with {max_steps} steps")
    sumsz_train = sum(nsample_train)
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )

    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )


    test_dist = np.array(test_data["distribution"]["test"])
    uni_dist = np.array([1/K]*K)
    # test_dist = uni_dist

    client_weight = np.array([0.5/N for n in range(N)])

    # client_weight = client_weight/np.sum(client_weight)

    x_test = tf.convert_to_tensor(test_data["user_data"]["test"]["x"])
    y_test = tf.convert_to_tensor(test_data["user_data"]["test"]["y"])
    test_batch_sz = B

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

        available_client = np.random.choice(N, r, replace=False)

        if "uniform" in policy:
            participants_set = client_sample_uni(available_client, m)
        elif "POC" in policy:
            participants_set = client_sample_POC(available_client, m, client_loss_train)
        elif "CBS" in policy:
            participants_set = client_sample_CBS(
                train_dist, available_client, m, nsample_train, t, client_cnt + 1, expfactor=param
            )
        elif "KL" in policy:
            participants_set, Aq, Yq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, client_weight,
                Aq, Yq, Zq,
                V=param, min_rate_lim=True, max_rate_lim=True
            )

        weights = []
        for n in participants_set:
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
                       local_model, x_train, y_train, stepsize, max_steps, weight=wdict
                    )
            else:
                history, local_model = client_train(local_model, x_train, y_train, stepsize, max_steps)

            client_loss_train[n] = history.history["loss"][-1]
            client_acc_train[n] = history.history["sparse_categorical_accuracy"][-1]
            
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
        
        model_pred = global_model.predict(x_test, verbose=0)
        y_pred = tf.convert_to_tensor([np.argmax(v) for v in model_pred])

        f1_test[t] = f1_score(y_test, y_pred, average="macro")
        
        policy_dist = np.matmul(client_cnt/((t+1)*m), train_dist)
        shift_test[t] = KL(test_dist, policy_dist)

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
                wait = 0  # Reset counter when improvement is significant
            else:
                wait += 1  # No significant improvement, increment counter

            # Trigger early stopping if no improvement for a set number of rounds
            if wait >= patience:
                print(f"\nEarly stopping triggered after {t + 1} rounds!")
                break  # Stop training
        
                
        stepsize = stepsize * decay_rate
        bkd.clear_session()

    del global_model, y_pred, model_pred
    gc.collect()
    bkd.clear_session()

    return train_log


def main():
    for param_list in run_list:
        print(f"Running version {v} with V = {param_list} and seed = {seed_list}")
        for rseed in seed_list:
            with concurrent.futures.ProcessPoolExecutor(initializer=init, initargs=(v, rseed), max_workers=nworkers, mp_context=multiprocessing.get_context("fork")) as executor:
                futures = [executor.submit(FedAvg, policy, param, rseed, True, False) for param in param_list]
                for future in concurrent.futures.as_completed(futures):
                    results = future.result()
                    policy_name = results.pop("policy_name")
                    print(f"{policy_name} completed")
                    with open(f"results/{dataset}_hypertune/{dataset}_train{train_group}_test{test_group}_{v}_{policy_name}.json", "w") as f:
                        json.dump(results, f)   


if __name__ == "__main__":
    main()
