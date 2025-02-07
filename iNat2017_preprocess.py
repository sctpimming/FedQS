import csv
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
import pickle
from numpy.random import dirichlet, choice
from tqdm import tqdm


train_path = "data/iNat-120k/inaturalist-user-120k/federated_train_user_120k.csv"
test_path = "data/iNat-120k/inaturalist-user-120k/test.csv"

# Create data structure
train_data = {"users": [], "user_data": {}, "distribution": {}, "num_samples": []}
test_data = {"users": [], "user_data": {}, "distribution": {}}

K = 1203

import subprocess
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

def get_path(imgid, path="data/iNat-120k/train_val_images/"):
    file_name = f"{imgid}.jpg"
    # for root, dirs, files in os.walk(path):
    #     if name in files:
    #         file_path = os.path.join(root, name)
    #         return file_path
    command = ['locate', file_name]

    output = subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0]
    output = output.decode()

    search_results = output.split('\n')
    file_path = search_results[0]
    return file_path


label_list = []
idx = 0
with open(train_path, "r") as data:
    reader = csv.reader(data)
    next(reader)
    for line in tqdm(reader):
        uname, imgid, label, _ = line
        label = int(label)
        if uname not in train_data["users"]:
            train_data["users"].append(uname)
            train_data["user_data"][uname] = {
                "x":[], "y":[]
            }
        train_data["user_data"][uname]["x"].append(get_path(imgid))
        train_data["user_data"][uname]["y"].append(label)


for uname in train_data["users"]:
    train_sz = len(train_data["user_data"][uname]["y"])
    train_data["num_samples"].append(train_sz)
    label_dist = np.zeros(K)
    for label in train_data["user_data"][uname]["y"]:
        label_dist[label] += 1
    label_dist = label_dist / train_sz
    train_data["distribution"][uname] = label_dist

# Testing dataset
X_test = []
Y_test = []

with open(test_path, "r") as data:
    reader = csv.reader(data)
    next(reader)
    for line in tqdm(reader):
        imgid, label, _ = line
        X_test.append(get_path(imgid))
        Y_test.append(int(label))



alpha_train = 0.3
alpha_test = 0.1

combined = list(zip(X_test, Y_test))
sorted_test = sorted(combined, key=lambda x: x[1])
grouped_test = [[v[0] for v in sorted_test if v[1] == k] for k in range(K)]

num_samples_test = int(len(Y_test) * 1)
print(num_samples_test)
distribution_test = dirichlet([alpha_test] * K, size=1)

uname = "test"

test_label_n = choice(K, num_samples_test, p=distribution_test[0])
test_n = [(grouped_test[k][choice(len(grouped_test[k]))], int(k)) for k in test_label_n]

# # # guarantee non-zero PMF
# for k in range(K):
#     test_n.append((grouped_test[k][choice(len(grouped_test[k]))], int(k)))
# test_n = combined
plot_img(test_n[0][0])
test_data["users"].append(uname)
test_data["user_data"][uname] = {
    "x": [v[0] for v in test_n],
    "y": [v[1] for v in test_n],
}
label_dist = np.zeros(K)
for v in test_n:
    label = v[1]
    label_dist[label] += 1
print(label_dist)
label_dist = label_dist / num_samples_test
print(label_dist)
test_data["distribution"][uname] = label_dist

train_path = f"data/Federated/iNat_train.pck"
test_path = f"data/Federated/iNat_test.pck"

with open(train_path, "wb") as f:
    pickle.dump(train_data, f)
with open(test_path, "wb") as f:
    pickle.dump(test_data, f)