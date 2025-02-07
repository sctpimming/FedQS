import keras
from keras import layers, ops
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def balanced_loss(weight):
  def bceloss(y_true, y_pred):
    celoss = keras.losses.SparseCategoricalCrossentropy()
    loss = celoss(y_true, y_pred, sample_weight=weight)
    return loss
  return bceloss

def prox_loss(prox_term):
    def celoss(y_true, y_pred):
        loss = ops.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1) + prox_term
        return loss
    return celoss

