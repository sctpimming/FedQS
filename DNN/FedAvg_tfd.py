import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
sys.path.append('code')
import matplotlib.pyplot as plt # Import matplotlib for plotting

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
import numpy as np
import tensorflow as tf
import keras
from keras import backend as bkd
from sklearn.metrics import f1_score, classification_report
import concurrent.futures
import gc
import time
import wandb

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
import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.config.run_functions_eagerly(True)

TARGET_IMAGE_SIZE = (299, 299)
TARGET_IMAGE_CHANNELS = 3
INPUT_SIZE = (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], TARGET_IMAGE_CHANNELS)

USING_GN = False
GROUP_NUM = None
GROUP_SIZE = 2

IS_TEST = False
SUBSAMPLE_SIZE = 3200

def get_metadata(dataset, rseed=0):
    train_path = f"data/{dataset}_train/train_metadata.json"
    if IS_TEST:
        test_path = f"data/{dataset}_test/test_metadata.json"
    else:
        test_path = f"data/{dataset}_val/val_metadata.json"
    
    with open(train_path, "rb") as f:
        train_metadata = json.load(f)
    with open(test_path, "rb") as f:
        test_metadata = json.load(f)
    
    return train_metadata, test_metadata

def load_tfrecord_dataset(
    tfrecord_file_paths: list,
    batch_size: int = 32,
    shuffle_buffer_size: int = 1000,
    is_training: bool = True,
    compression_type: str = "GZIP"
):
    if not isinstance(tfrecord_file_paths, list):
        tfrecord_file_paths = [tfrecord_file_paths]

    for path in tfrecord_file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"TFRecord file not found: {path}")

    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_and_preprocess_example(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=TARGET_IMAGE_CHANNELS)
        image = tf.cast(image, tf.float32) # Image now in [0, 255] float32 range

        # --- AUGMENTATION LOGIC ---
        if is_training:
            original_shape = tf.shape(image)
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

            sampled_box_begin, sampled_box_size, _ = tf.image.sample_distorted_bounding_box(
                image_size=original_shape,
                bounding_boxes=bbox,
                min_object_covered=0.08,
                aspect_ratio_range=(3/4, 4/3),
                area_range=(0.08, 1.0),
                max_attempts=100,
            )
            image = tf.slice(image, sampled_box_begin, sampled_box_size)
            image = tf.image.resize(image, TARGET_IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, TARGET_IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)

        # --- NORMALIZATION STEP ---
        # Use MobileNetV2's specific preprocess_input function
        image = mobilenet_v2_preprocess_input(image) # This handles scaling from [0, 255] to [-1, 1]

        label = parsed_features['label']
        label = tf.ensure_shape(label, [])

        return image, label

    dataset = tf.data.TFRecordDataset(tfrecord_file_paths, 
                                      compression_type=compression_type, 
                                      num_parallel_reads=tf.data.AUTOTUNE)

    dataset = dataset.map(_parse_and_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training and shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)    
        dataset = dataset.repeat()    
    if not is_training:
        dataset = dataset.take(SUBSAMPLE_SIZE)
    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def set_gpu_memory_growth(ram_limit=None):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enforce memory growth on all visible GPUs (which should only be one)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # This occurs if you try to set memory growth after devices have been used
            print(f"TensorFlow GPU setup error: {e}")

def client_selector(policy, metric, params=None):
    queue_backlog = None
    recursive_weight = None
    if "uniform" in policy:
        participants_set = client_sample_uni(metric)
    elif "POC" in policy:
        participants_set = client_sample_POC(metric, dmult=params)
    elif "CBS" in policy:
        participants_set = client_sample_CBS(metric, expfactor=params)
    elif "ODFL" in policy:
        participants_set, recursive_weight = client_sample_ODFL(metric)
    elif "KL" in policy:
        participants_set, queue_backlog= client_sample_KL(
            metric, V=params, min_rate_lim=True, max_rate_lim=False
        )
    return participants_set, queue_backlog, recursive_weight

def client_train(dataset, client_idx, global_weights, stepsize, nsteps, B, K, nsample, weight=None):
    local_model = models.get_MobileNet(
        K = K,
        input_shape=INPUT_SIZE,
        IsGN=USING_GN,
        group_sz=GROUP_SIZE,
        group_num=GROUP_NUM, 
        learning_rate=stepsize
    )
    local_model.set_weights(global_weights) 
    
    print(f"\nTraining client {client_idx} with {nsample} samples")
    file_path = f"data/{dataset}_train/user_{client_idx}.tfrecord"

    train_dataset = load_tfrecord_dataset(file_path, batch_size=B, shuffle_buffer_size=nsample , compression_type=None)

    history = local_model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=nsteps,
        verbose=2,
        class_weight = weight,
    )
    local_weight = local_model.get_weights()

    del train_dataset
    del local_model

    return (client_idx, history, local_weight)

def model_evaluation(dataset, global_model, B):
    if IS_TEST:
        file_path = f"data/{dataset}_test/{dataset}_test_set.tfrecord"
    else:
        file_path = f"data/{dataset}_val/{dataset}_val_set.tfrecord"
    test_dataset = load_tfrecord_dataset(file_path, batch_size=B, is_training=False, compression_type=None)

    loss, acc= global_model.evaluate(test_dataset, verbose=2)

    return loss, acc
    
def FedAvg_tfd():
    with wandb.init() as run:
        config = run.config
        set_gpu_memory_growth()
        
        if config.batch_size == 16:
            nworkers = 3
        else:
            nworkers = 5

        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=nworkers,
            initializer=set_gpu_memory_growth,  
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1
        )

        train_metadata, test_metadata = get_metadata(config.dataset_name)
        user_name = list(train_metadata.keys())

        N = len(user_name)
        r = int(N*0.3)
        m = config.n_participants
        T = config.epoch

        train_dist = np.array(
            [train_metadata[uname]["label_distribution"] for uname in user_name]
        )
        nsample_train = [train_metadata[uname]["num_examples"] for uname in user_name]

        szfrac_train = np.array(
            [nsample_train[n] / sum(nsample_train) for n in range(N)]
        )
        
        if IS_TEST:
            test_dist = np.array(test_metadata["test_set"]["label_distribution"])
        else:
            test_dist = np.array(test_metadata["val_set"]["label_distribution"])

        K = len(test_dist)
        
        print(f"Algorithm: {config.policy_name}")
        print(f"There is {N} clients and {K} classes")

        np.random.seed(0)
        policy_name = config.policy_name
        
        decay_rate = 1
        stepsize = config.learning_rate

        queues = {
            "Aq": np.zeros(N),
            "Yq": np.zeros(N),
            "Zq": np.zeros(N)
        }

        metrics = {
            "loss_train": np.zeros(T),
            "loss_test": np.zeros(T),
            "acc_train": np.zeros(T),
            "acc_test": np.zeros(T),
            "shift_test": np.zeros(T),
            "client_cnt": np.zeros(N),
            "client_loss": np.ones(N) * 1000,
            "client_acc": np.zeros(N)
        }

        global_model = models.get_MobileNet(
            K = K,
            input_shape=INPUT_SIZE,
            IsGN=USING_GN,
            group_sz=GROUP_SIZE,
            group_num=GROUP_NUM, 
            learning_rate=stepsize
        )

        for t in tqdm(range(T)):
            available_client = np.random.choice(N, r, replace=False)
            train_info = {
                "round": t+1,
                "train_dist":train_dist,
                "test_dist":test_dist,
                "available_client": available_client,
                "n_participants": m,
                "client_sample": nsample_train,
                "sample_frac": szfrac_train,
                "client_count": metrics["client_cnt"] + 1,
                "client_loss": metrics["client_loss"],
                "queue_backlog": (queues["Aq"], queues["Yq"], queues["Zq"])
            }

            participants_set, queue_backlog, recursive_weight = client_selector(policy_name, train_info, config.params_value)
            if queue_backlog is not None:    
                Aq, Yq, Zq = queue_backlog
                queues["Aq"] = Aq
                queues["Yq"] = Yq
                queues["Zq"] = Zq


            futures = []
            global_weights = global_model.get_weights()
            new_weights = [np.zeros_like(w, dtype=np.float32) for w in global_weights]
            participants_nsample = sum([nsample_train[n] for n in participants_set])
            if "ODFL" in policy_name:
                total_weight = sum([recursive_weight[idx] for idx in participants_set])
                weight_coefficient = np.zeros(N)
                for idx in participants_set:
                    weight_coefficient[idx] = recursive_weight[idx] / total_weight 

            for n in participants_set:
                metrics["client_cnt"][n] += 1
                wdict = {

                        k: (
                            float(test_dist[k] / train_dist[n][k])
                            if train_dist[n][k] > 0
                            else 0
                        )
                        for k in range(K)
                }

                if "balanced" in policy_name:
                    class_weight = wdict
                else: 
                    class_weight = None
                
                futures.append(
                    executor.submit(
                        client_train, 
                        config.dataset_name, 
                        n, 
                        global_weights, 
                        stepsize, 
                        config.n_steps, 
                        config.batch_size, 
                        K, 
                        nsample_train[n], 
                        class_weight
                    )
                )

            
            for future in concurrent.futures.as_completed(futures):
                try:
                    client_idx, history, local_weight = future.result()
                    if "ODFL" in policy_name:
                        weight_factor = weight_coefficient[client_idx]
                    elif "CBS" in policy_name:
                        weight_factor = (nsample_train[client_idx]/participants_nsample)
                    else:
                        weight_factor = 1/m

                

                    # Incrementally update the new_weights
                    for i, layer_weights in enumerate(local_weight):
                        new_weights[i] += layer_weights * weight_factor

                    metrics["client_loss"][client_idx] = history.history["loss"][-1]
                    metrics["client_acc"][client_idx] = history.history["sparse_categorical_accuracy"][-1]
                except Exception as e:
                    print(f"--- WORKER EXCEPTION DETECTED: {e} ---")
                    return  # <--- Stops the current W&B run immediately
            
        
            global_model.set_weights(new_weights)
            
            
            loss, acc= model_evaluation(config.dataset_name, global_model, B=config.batch_size)

            # Update metric 

            metrics["loss_test"][t] = loss
            metrics["acc_test"][t] = acc
            
            policy_dist = np.matmul(metrics["client_cnt"]/((t+1)*m), train_dist)
            metrics["shift_test"][t] = KL(test_dist, policy_dist)

            masked = (metrics["client_loss"] < 1000)
            metrics["loss_train"][t] = sum(metrics["client_loss"][masked])/max(sum(masked), 1)
            metrics["acc_train"][t] = np.dot(metrics["client_acc"], szfrac_train)

            wandb.log({
                "average_train_loss": metrics["loss_train"][t],
                "average_train_acc": metrics["acc_train"][t],
                "global_test_loss": metrics["loss_test"][t],
                "global_test_accuracy": metrics["acc_test"][t]
            }, step=t)
                    
            stepsize = stepsize * decay_rate

        if IS_TEST:
            wandb.log({
                'final_epoch': t,
                'metrics': metrics,
                'queues_backlog': queues,
                'policy_name': policy_name
            })

