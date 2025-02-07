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

def sparse_weighted_loss(class_weights):
    """
    Custom sparse loss function with class weights.

    Args:
        class_weights (dict): A dictionary where keys are class indices and values are the corresponding weights.

    Returns:
        A loss function that incorporates class weights.
    """
    def loss_fn(y_true, y_pred):
        # Convert class weights dictionary to a tensor
        weights = tf.constant([class_weights[i] for i in range(len(class_weights))], dtype=tf.float32)
        print(weights)
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Gather the weights for each class from y_true
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)  # Ensure y_true is the right shape and type
        sample_weights = tf.gather(weights, y_true)  # Get weights for each sample based on true class

        # Compute the sparse categorical crossentropy
        cce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        print(cce)
        # Multiply the loss by the corresponding weights
        weighted_cce = cce * sample_weights

        # Return the mean weighted loss
        return tf.reduce_mean(weighted_cce)
    
    return loss_fn

# Dummy inputs
y_true = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)  # One-hot labels
y_pred = tf.constant([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=tf.float32)  # Predicted probabilities
class_weights = tf.constant([1.0, 2.0, 1.5], dtype=tf.float32)  # Class weights

# Test the loss function
loss = sparse_weighted_loss(y_true, y_pred, class_weights)
print("Loss:", loss)