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

N = 5000
r = int(0.3 * N)
m = 5
B = 32
I = 5
K = 200
T = 1000

dataset = "Tinyimagenet"
train_group = "G1"
test_group = "G1"
v = "test1"

seed_list = [0]
run_list = [0.1]
policy_list = ["uniform_balanced", "POC_balanced", "CBS_balanced", "KL_balanced"]
nworkers = 2

def get_client_dataset(uname):
    client_dataset_dir = f"{dataset_dir}/{uname}/"
    batch_size = B
    if uname == "test":
        batch_size = 64

    dataset = tf.data.Dataset.load(client_dataset_dir, compression="GZIP")
    dataset_size = sum(1 for _ in dataset)

    dataset = dataset.shuffle(buffer_size=dataset_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def get_metadata():
    with open(dataset_dir+"train_data.pck", "rb") as f:
        train_data = pickle.load(f)
    with open(dataset_dir+"test_data.pck", "rb") as f:
        test_data = pickle.load(f)
    return train_data, test_data

def load_checkpoint(policy_name):
    model_path = f'./model_checkpoint/{policy_name}.keras'
    metadata_path =  f'./model_checkpoint/{policy_name}_metadata.json'
    loaded_model = tf.keras.models.load_model(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return loaded_model, metadata

def save_checkpoint(model, metadata, policy_name):
    model_path = f'./model_checkpoint/{policy_name}.keras'
    metadata_path =  f'./model_checkpoint/{policy_name}_metadata.json'

    model.save(model_path)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    print(f"Checkpoint saved at epoch {metadata['epoch']}")

def init(v):
    global dataset_dir
    dataset_dir = f"data/{dataset}_train{train_group}_test{test_group}_{v}/"
    gpus = tf.config.list_physical_devices('GPU')
    print(len(gpus))
    if len(gpus) > 0:
        # tf.config.set_logical_device_configuration(
        # gpus[0],
        # [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4)])
        tf.config.experimental.set_memory_growth(gpus[0], True)
    
def client_train(model, uname, stepsize, max_steps, weight=None):
    model.optimizer.learning_rate.assign(stepsize)
    client_dataset = get_client_dataset(uname)
    history = model.fit(
        client_dataset,
        epochs=max_steps,
        steps_per_epoch = 1,
        verbose=0,
        class_weight = weight,
    )

    del client_dataset
    gc.collect()

    return history, model

def FedAvg(policy, param, rseed, from_ckt=False, Debug=True, early_stop=False):
    print(f"Algorithm: {policy} Random Seed: {rseed}")
    np.random.seed(int(rseed))
    tf.random.set_seed(int(rseed))

    policy_name = policy
    patience = 100
    min_delta = 0.001
    wait = 0

    decay_rate = 1

    if from_ckt:
        saved_model, metadata = load_checkpoint(policy_name)
        epoch = np.array(metadata["epoch"])
        print(f"Checkpoint loaded at epoch {metadata['epoch']}")
        best_accuracy = metadata["best_accuracy"]
        stepsize = metadata["stepsize"]

        Aq = np.array(metadata["Aq"])
        Yq = np.array(metadata["Yq"])
        Zq = np.array(metadata["Zq"])

        loss_train = np.array(metadata["loss_train"])
        loss_test = np.array(metadata["loss_test"])
        acc_train = np.array(metadata["acc_train"])
        acc_test = np.array(metadata["acc_test"])
        f1_test = np.array(metadata["f1_test"])
        shift_test = np.array(metadata["shift"])

        client_cnt = np.array(metadata["client_cnt"])
        client_loss_train = np.array(metadata["client_loss_train"])
        client_acc_train = np.array(metadata["client_acc_train"])
    else:
        epoch = 0
        best_accuracy = 0.00
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
    train_data, test_data = get_metadata()
    train_dist = np.array(
        [train_data[uname]["distribution"] for uname in user_name]
    )

    nsample_train = np.array(
        [train_data[uname]["num_samples"] for uname in user_name]
    )
    max_steps = int(max(nsample_train)/B) * I
    print(f"Training with {max_steps} steps")
    sumsz_train = sum(nsample_train)
    szfrac_train = nsample_train/sumsz_train
    epsilon = 1e-3
    for n in range(N):
        train_dist[n] = np.array([p if p > 0 else epsilon for p in train_dist[n]])
        train_dist[n] = train_dist[n]/sum(train_dist[n])

    test_dataset = get_client_dataset(uname="test")
    test_dist = np.array(test_data["test"]["distribution"])

    if from_ckt:
        global_model = saved_model
        local_model = models.get_ResNet50(weight_config=None, K=200, rseed=rseed)
    else:
        global_model = models.get_ResNet50(weight_config=None, K=200, rseed=rseed)
        local_model = models.get_ResNet50(weight_config=None, K=200, rseed=rseed)
    for t in tqdm(range(epoch, T)):
        bkd.clear_session()

        global_weights = [tf.zeros_like(w) for w in global_model.get_weights()]

        masked = np.isfinite(client_loss_train)
        loss_train[t] = sum(client_loss_train[masked])/max(sum(masked), 1)
        acc_train[t] = np.dot(client_acc_train, szfrac_train)

        available_client = np.random.choice(N, r, replace=False, p=szfrac_train)
        
        if "uniform" in policy:
            participants_set = client_sample_uni(available_client, m)
        elif "CBS" in policy:
            participants_set = client_sample_CBS(
                train_dist, available_client, m, nsample_train, t, client_cnt + 1, expfactor=10
            )
        elif "KL" in policy:
            participants_set, Aq, Yq, Zq = client_sample_KL(
                train_dist, test_dist, available_client, m, szfrac_train,
                Aq, Yq, Zq, 
                V=param, 
                min_rate_lim=True, max_rate_lim=True
            )
        elif "POC" in policy:
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
                    local_model, user_name[n], stepsize, max_steps, weight=wdict
                )
            else:
                history, local_model = client_train(local_model, user_name[n], stepsize, max_steps)

            client_loss_train[n] = history.history["loss"][-1]
            client_acc_train[n] = history.history["sparse_categorical_accuracy"][-1]

            client_weights = local_model.get_weights()
            client_weights = [tf.convert_to_tensor(w) for w in client_weights]
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
        
        metadata = {
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

        if t%50 == 0:
            save_checkpoint(global_model, metadata, policy_name)

        if early_stop:
            if acc_test[t] - best_accuracy > min_delta:
                best_accuracy = acc_test[t]
                wait = 0  # Reset counter when improvement is significant
            else:
                wait += 1  # No significant improvement, increment counter

            # Trigger early stopping if no improvement for a set number of rounds
            if wait >= patience:
                print(f"\nEarly stopping triggered after {t + 1} rounds!")
                break  # Stop training

        del global_weights
        gc.collect()
        bkd.clear_session()
            
    
    del global_model
    gc.collect()
    bkd.clear_session()

    return metadata

def main():
    n_attempts = 10
    for param_val in run_list:
        print(f"Running version {v} with the following seed {seed_list}")
        for rseed in seed_list:
            load_ckt = False
            for att in range(n_attempts):
                print(f"Trying attempts {att}")
                try:
                    with concurrent.futures.ProcessPoolExecutor(initializer=init, initargs=(v, ), max_workers=nworkers, mp_context=multiprocessing.get_context("fork")) as executor:
                        futures = [executor.submit(FedAvg, policy, param_val, rseed, load_ckt) for policy in policy_list]
                        for future in concurrent.futures.as_completed(futures):
                            results = future.result()
                            policy_name = results.pop("policy_name")
                            print(f"{policy_name} completed")
                            with open(f"results/{dataset}_results/{dataset}_train{train_group}_{v}_test{test_group}_{policy_name}.json", "w") as f:
                                json.dump(results, f) 
                    break
                except Exception as e:
                    print(e)
                    load_ckt = True


if __name__ == "__main__":
    main()
