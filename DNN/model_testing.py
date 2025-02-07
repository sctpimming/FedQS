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



import pickle
from tqdm import tqdm
import models

def import_data(v):
    train_path = f"data/Federated/iNat_train.pck"
    test_path = f"data/Federated/iNat_train.pck"
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

init("v1")
global_model = keras.models.load_model('iNat_QS_checkpoint.keras')
user_name = [v for v in train_data["users"]]

for n in range(len(user_name)):
    x_test = test_data["user_data"][user_name[n]]["x"]
    y_test = test_data["user_data"][user_name[n]]["y"]

    test_batch_sz = 32
    x_test_batch, y_test_batch = get_batch(x_test, y_test, batch_size=test_batch_sz)
    test_scores = global_model.evaluate(x_test_batch, y_test_batch, verbose=0, batch_size=test_batch_sz)
    # predictions = global_model.predict(x_test_batch)
    print(test_scores)
    # print([((-predictions[i]).argsort()[:5], y_test_batch[i]) for i in range(test_batch_sz)])
