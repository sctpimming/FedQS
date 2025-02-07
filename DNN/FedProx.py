import os


# This guide can only be run with the TensorFlow backend.
os.environ["KERAS_BACKEND"] = "tensorflow"


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import sys
sys.path.append('code')
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf
import keras
from keras import backend as bkd
from sklearn.metrics import f1_score
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
    client_sample_QS,
    client_sample_KL,
)
import models
#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def import_data(v):
    train_path = f"data/Federated/CIFAR_alpha10_{v}_train.pck"
    test_path = f"data/Federated/CIFAR_alpha10_{v}_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def init(v):
    global train_data
    global test_data
    train_data, test_data = import_data(v)



def client_train(model, global_weights, x_train, y_train, stepsize,  mu =0.01):
    optimizer = keras.optimizers.SGD(learning_rate=stepsize)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(B)

    i = 0
    while i < I:
        for (x_batch_train, y_batch_train) in train_dataset:
            local_weights = model.get_weights()
            i += 1
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                prox_term = 0
                # print(local_weights.shape, global_weights.shape)
                for local_w, global_w in zip(local_weights, global_weights):
                    # print(type(local_w))
                    local_w = local_w.reshape(-1)
                    global_w = global_w.reshape(-1)
                    # print(local_w.shape, global_w.shape)
                    prox_term += np.linalg.norm(local_w- global_w)**2
                    # print(np.linalg.norm(local_w- global_w)**2)
                print(f"The value of proximal term is: {((mu/2) * prox_term)}")
                loss_value = loss_fn(y_batch_train, logits) + ((mu/2) * prox_term)
                print(f"loss: {loss_value}")

            grads = tape.gradient(loss_value, model.trainable_weights)

            optimizer.apply(grads, model.trainable_weights)
    return model

np.random.seed(12345)

N = 100
r = 60
m = 9
B = 50
I = 64
K = 10
T = 500


def FedAvg(policy, Debug=True):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*2)])
    Aq = np.zeros(N)
    Zq = np.zeros(N)

    
    loss_train = np.zeros(T)
    loss_test = np.zeros(T)
    acc_train = np.zeros(T)
    acc_test = np.zeros(T)
    f1_test = np.zeros(T)
    client_cnt = np.zeros(N)

    user_name = ["f_{0:05d}".format(n) for n in range(N)]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    sumsz_train = sum([train_data["num_samples"][n] for n in range(N)])
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    # wdict =  {k:test_dist[k] for k in range(K)}

    x_test = tf.convert_to_tensor(test_data["user_data"]["test"]["x"])
    y_test = tf.convert_to_tensor(test_data["user_data"]["test"]["y"])
    test_batch_sz = y_test.shape[0]

    decay_rate = 0.9992
    stepsize = 0.005

    global_model = models.get_ChoNet(config="FedProx")
    local_model = models.get_ChoNet(config="FedProx")
    
    for t in tqdm(range(T)):
        keras.backend.clear_session()
        client_eval = [
            global_model.evaluate(
                np.array(train_data["user_data"][user_name[n]]["x"]),
                np.array(train_data["user_data"][user_name[n]]["y"]),
                verbose=0,
                batch_size=32
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
            participants_set = client_sample_uni(available_client, m, szfrac_train)
        elif policy == "CBS":
            participants_set = client_sample_CBS(
                train_dist, available_client, m, B, t, client_cnt + 1
            )
        elif policy == "KL" or policy == "KL_balanced":
            participants_set, Aq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, Aq, Zq, V=100
            )
        elif policy == "POC":
            participants_set = client_sample_POC(available_client, m, client_loss_train)

        weights = []
        for n in participants_set:
            keras.backend.clear_session()
            client_cnt[n] += 1

            local_model.set_weights(global_model.get_weights())

            x_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["x"])
            y_train = tf.convert_to_tensor(train_data["user_data"][user_name[n]]["y"])

            local_model = client_train(local_model, global_model.get_weights(), x_train, y_train, stepsize)
            weights.append(local_model.get_weights())


        new_weights = list()

        for weights_list_tuple in zip(*weights): 
            new_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
            
        del weights
        global_model.set_weights(new_weights)

    
        test_scores = global_model.evaluate(x_test, y_test, verbose=0, batch_size=test_batch_sz)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]
        
        if Debug:
            print(f'{policy}:{loss_train[t]:.4f}', f'{acc_train[t]:.4f}', f'{loss_test[t]:.4f}', f'{acc_test[t]:.4f}')

        # model_pred = global_model.predict(x_test, verbose=0)
        # y_pred = [np.argmax(v) for v in model_pred]
        # f1_test[t] = f1_score(y_test, y_pred, average="macro")

        #stepsize = stepsize * decay_rate
        if t == 150 or t == 300:
            stepsize=stepsize/2
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
        policy
    )


def main():
    # multiprocessing.set_start_method('fork')/
    # print(tf.test.is_gpu_available)
    run_list = ["v1"]
    for v in run_list:
        print(f"Running version {v}")
                  #init(v)
        policy_list = ["uniform"]
        with concurrent.futures.ProcessPoolExecutor(initializer=init, initargs=(v, ), max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(FedAvg, policy) for policy in policy_list]
            for future in concurrent.futures.as_completed(futures):
                results = future.result()
                policy_name = results[-1]
                results = results[:-1]
                print(f"\n {policy_name} completed")
                res_dict= {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[]}
                for i, metric in enumerate(res_dict):
                    res_dict[metric] = results[i]
                with open(f"results/CIFAR_alpha10_{v}_FedProx.json", "w") as f:
                    json.dump(res_dict, f)  


if __name__ == "__main__":
    main()
