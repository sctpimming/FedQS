import os

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
from sklearn.metrics import f1_score
from PIL import Image
import gc
from matplotlib import pyplot as plt


import pickle
from tqdm import tqdm
import models

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

def get_img(file_path):
    img = Image.open(file_path)
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    img = np.array(img)/255.0
    if img.shape[-1] == 4:
        print(file_path)
    img = tf.convert_to_tensor(img)
    if img.ndim < 3:
        img = tf.image.grayscale_to_rgb(tf.expand_dims(img, -1))
    img = tf.image.resize(img, size=(224, 224))
    return img

def get_batch(x, y, batch_size=50):
    batch_idx = np.random.choice(len(y), batch_size) 
    x_batch = tf.convert_to_tensor([get_img(x[idx]) for idx in batch_idx])
    y_batch = tf.convert_to_tensor([y[idx] for idx in batch_idx])
    return x_batch, y_batch


def plot_img(file_path):
    img = Image.open(file_path)
    if img.mode == 'CMYK':
        img = img.convert('RGB')
    img = np.array(img)/255.0
    if img.shape[-1] == 4:
        print(file_path)
    img = tf.convert_to_tensor(img)
    if img.ndim < 3:
        img = tf.image.grayscale_to_rgb(tf.expand_dims(img, -1))
    img = np.array(tf.image.resize(img, size=(224, 224)))
    plt.imshow(img, interpolation='nearest')
    plt.show()

init("v1")
B = 16
user_name = [v for v in train_data["users"]]
x_train = train_data["user_data"][user_name[0]]["x"]
y_train = tf.convert_to_tensor(train_data["user_data"][user_name[0]]["y"])
x_batch, y_batch = get_batch(x_train, y_train, batch_size=B)
x_test = test_data["user_data"]["test"]["x"]
y_test = test_data["user_data"]["test"]["y"]

for i in range(B):
    x = x_batch[i]
    y = y_batch[i]
    test_img = get_img(x_test[y_test.index(y)])
    print(x.dtype, test_img.dtype)
    plt.subplot(1, 2, 1)
    plt.imshow(x, interpolation='nearest')
    plt.subplot(1, 2, 2)
    plt.imshow(test_img, interpolation='nearest')
    plt.show()
