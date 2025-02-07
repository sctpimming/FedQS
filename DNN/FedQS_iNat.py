import os
import copy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import sys
sys.path.append('code')
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import numpy as np
import tensorflow as tf
import keras
from keras import backend as bkd
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import gc


import json
import multiprocessing
import concurrent.futures
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
    train_path = f"data/Federated/iNat_train.pck"
    test_path = f"data/Federated/iNat_test.pck"
    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def init(v):    
    global train_data
    global test_data
    train_data, test_data = import_data(v)

def get_img(file_path, is_test=False):
    img = Image.open(file_path)
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    img = np.array(img)
    if img.shape[-1] == 4:
        print(file_path)
    img = tf.convert_to_tensor(img)
    if img.ndim < 3:
        img = tf.image.grayscale_to_rgb(tf.expand_dims(img, -1))
    img_h = img.shape[0]
    img_w = img.shape[1] 
    if not is_test:
        params = np.random.uniform(0.5, 1.5, 3)
        # img = tf.image.adjust_contrast(img, params[0])
        # img = tf.image.random_brightness(img, params[1])
        # img = tf.image.random_hue(img, params[2])
        # img = tf.image.random_crop(img, size=(min(200, img_h), min(200, img_w), 3))
    img = tf.image.resize(img, size=(224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(
            img, data_format=None
            )
    # print(np.min(img), np.max(img))
    return img

def get_batch(x, y, batch_size=64, is_test=False):
    if batch_size > len(y):
        batch_size = len(y)
    batch_idx = np.random.choice(len(y), batch_size, replace=False) 
    x_batch = tf.convert_to_tensor([get_img(x[idx], is_test) for idx in batch_idx])
    y_batch = tf.convert_to_tensor([y[idx] for idx in batch_idx])
    return x_batch, y_batch

def client_train(model, x_train, y_train, stepsize, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    for _ in range(I):
        x_batch, y_batch = get_batch(x_train, y_train)
        B = min(64, len(y_batch))
        history = model.fit(
            x_batch,
            y_batch,
            batch_size=B,
            epochs=1,
            steps_per_epoch=1,
            shuffle=True,
            verbose=0,
        )
        # scores = model.evaluate(x_batch, y_batch, verbose=1, batch_size=B)
        # print(scores)
    del x_batch, y_batch
    gc.collect()
    return history, model


N = 9275
r = 100
m = 10
B = 32
I = 10
K = 1203
T = 1000


def FedAvg(policy, rseed, start=0, Debug=True):
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Algorithm: {policy} Random Seed: {rseed}")
    np.random.seed(rseed)
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024*24)])
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

    user_name = [v for v in train_data["users"]]
    train_dist = np.array(
        [train_data["distribution"][uname] for uname in train_data["distribution"]]
    )
    sumsz_train = sum([train_data["num_samples"][n] for n in range(N)])
    print(sumsz_train)
    szfrac_train = np.array(
        [train_data["num_samples"][n] / sumsz_train for n in range(N)]
    )
    test_dist = np.array(test_data["distribution"]["test"])
    nsample_train = [train_data["num_samples"][n] for n in range(N)]
    Q_inv = np.array([1/q if q > 0 else 10 ** (-20) for q in test_dist])

    # wdict =  {k:test_dist[k] for k in range(K)}

    x_test = test_data["user_data"]["test"]["x"]
    y_test = test_data["user_data"]["test"]["y"]
    test_batch_sz = 64

    
    
    decay_rate = 1
    stepsize = 0.01

    if policy == "prox":
        global_model = models.get_MobileNet(config="prox")
    else:
        global_model = models.get_MobileNet()
        local_model = models.get_MobileNet()
    
    if start > 0:
        pass
    for t in tqdm(range(start, T)):
        # keras.backend.clear_session()
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
                train_dist, available_client, m, nsample_train, t, client_cnt + 1
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
            # keras.backend.clear_session()            
            client_cnt[n] += 1
            local_model.set_weights(global_model.get_weights())

            x_train = train_data["user_data"][user_name[n]]["x"]
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
        
        x_test_batch, y_test_batch = get_batch(x_test, y_test, batch_size=test_batch_sz)
        test_scores = global_model.evaluate(x_test_batch, y_test_batch, verbose=0, batch_size=test_batch_sz)
        loss_test[t] = test_scores[0]
        acc_test[t] = test_scores[1]
        

        model_pred = global_model.predict(x_test_batch, verbose=0)
        y_pred = [np.argmax(v) for v in model_pred]

        f1_test[t] = f1_score(y_test_batch, y_pred, average="macro")
        
        policy_dist = np.matmul(client_cnt/((t+1)*m), train_dist)
        shift_test[t] = KL(test_dist, policy_dist)
        if acc_test[t] >= max(acc_test):
            report = classification_report(y_test_batch, y_pred, zero_division=0.0, output_dict=True)

        if Debug:
            print(f'{policy}: train loss:{loss_train[t]:.4f}, train acc: {acc_train[t]:.4f}, test loss: {loss_test[t]:.4f}, test acc: {acc_test[t]:.4f}, test macro f1: {f1_test[t]:.4f}, gradient norm: {gradient_norm[t]:.4f}')


        if t%5 == 0:
            global_model.save("iNat_checkpoint.keras")
            print(f"model is saved at epoch {t}")


        del x_test_batch, y_test_batch
        stepsize = stepsize * decay_rate
        gc.collect()

    del global_model
    gc.collect()
    tf.keras.backend.clear_session()

    return (
        list(loss_train),
        list(loss_test),
        list(acc_train),
        list(acc_test),
        list(client_cnt / T),
        list(f1_test),
        report,
        list(gradient_norm),
        list(shift_test),
        policy
    )


def main():
    # multiprocessing.set_start_method('fork')
    rseed = 0
    run_list = ["v1"]
    for v in run_list:
        print(f"Running version {v}")
        policy_list = ["balanced"]
        init(v)
        for policy in policy_list:
            results = FedAvg(policy, rseed)
            if "KL" in results[-1]:
                policy_name = f"{results[-1]}_V1_R075_maxlim"
            else:
                policy_name = results[-1]
            results = results[:-1]
            print(f"\n {policy_name} completed")
            res_dict= {"loss_train":[], "loss_test":[], "acc_train":[], "acc_test":[], "client_cnt":[], "f1_test":[], "report":{}, "gradient_norm":[], "shift":[]}
            for i, metric in enumerate(res_dict):
                res_dict[metric] = results[i]
            with open(f"results/iNat_alpha01_{v}_S{rseed}_{policy_name}.json", "w") as f:
                json.dump(res_dict, f)  

if __name__ == "__main__":
    main()
