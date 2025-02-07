import os
import shutil
import tensorflow as tf
import numpy as np
import keras
from numpy.random import dirichlet, choice
from util.misc import KL
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import gc


def restructure_tiny_imagenet(base_dir):
    """
    Restructure Tiny ImageNet directory where images are in subfolders.
    Moves all JPEG files from `class_folder/images/` to `class_folder/`.
    """
    class_folders = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for class_folder in class_folders:
        images_folder = os.path.join(class_folder, "images")

        if os.path.exists(images_folder):
            # Move each image from `images/` to `class_folder/`
            for img_file in os.listdir(images_folder):
                src_path = os.path.join(images_folder, img_file)
                dst_path = os.path.join(class_folder, img_file)
                shutil.move(src_path, dst_path)

            # Remove the now-empty `images` folder
            os.rmdir(images_folder)


def save_client_datasets(client_ds, uname, base_dir):
    client_dir = base_dir+f"{uname}"
    #for img, label in client_ds:
        #plt.imshow(img)
        #plt.show()
        #break
    tf.data.Dataset.save(
        client_ds, client_dir, compression="GZIP", shard_func=None, checkpoint_args=None
    )


def show_hist(train_dist, test_dist):
    K = len(test_dist)
    class_name = [f"{n}" for n in range(K)]
    bar_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:grey",
        "tab:olive",
        "tab:cyan",
    ]
    ylim = max(test_dist)
    plt.subplot(1, 2, 1)
    plt.bar(class_name, train_dist, color=bar_colors, width=0.9)
    plt.ylim([0, ylim])

    plt.subplot(1, 2, 2)
    plt.bar(class_name, test_dist, color=bar_colors, width=0.9)
    plt.ylim([0, ylim])

    plt.show()

def noniid_partition(v, dataset, alpha_train=0.5, alpha_test=0.5, train_ratio=0.8):
    base_dir = f"data/Tinyimagenet_alpha05_{v}/"
    N = 10
    dataset_list = list(dataset.as_numpy_iterator())
    num_samples = len(dataset_list)
    uname_list = ["f_{0:05d}".format(n) for n in range(N)]

    num_samples_train = np.random.lognormal(4, 2, (N)).astype(int)
    # num_samples_train = np.clip(num_samples_train, 32, 2048)
    num_samples_train = np.clip(num_samples_train, 10000, 12000)
    #64 -> 2048
    print(num_samples_train, sum(num_samples_train))
    num_samples_test = 10000

    train_data = {
        key:{"distribution":[], "num_samples":0}
        for key in uname_list
    }


    test_data = {
        "test":{"distribution":[], "num_samples":0}
    }



    grouped_dataset = {}
    grouped_train = {}
    grouped_test = {}
    for img, label in tqdm(dataset):
        label = int(label)
        if label not in grouped_dataset:
            grouped_dataset[label] = []
        # x = keras.ops.cast(img, "float32")
        # x = keras.applications.resnet50.preprocess_input(x)
        grouped_dataset[label].append((img/255.0, label))


    class_keys = list(grouped_dataset.keys())
    K = len(class_keys)

    for label in range(K):
        train_samples = int(len(grouped_dataset[label])*train_ratio)
        grouped_train[label] = grouped_dataset[label][:train_samples]
        grouped_test[label] = grouped_dataset[label][train_samples:]

    print("Train/Test split completed")
    print(f"There are {len(list(grouped_train.keys()))} classes in training dataset")
    print(f"There are {len(list(grouped_test.keys()))} classes in testing dataset")

    distribution_train = dirichlet([alpha_train] * K, size=N)
    distribution_test = dirichlet([alpha_test] * K, size=1)
    if v == "control":
        distribution_test = [[1/K]*K]

    for n in tqdm(range(N)):
        uname = "f_{0:05d}".format(n)
        train_label_n = choice(K, num_samples_train[n], p=distribution_train[n])
        freq = np.zeros(K)
        for k in train_label_n:
            freq[k] += 1
        sample_idx_list = [
            choice(len(grouped_train[k]), min(int(freq[k]), 400), replace=False) 
            for k in range(K)
        ]         
        x_n = [
            tf.squeeze(grouped_train[k][idx][0]) 
            for k in range(K)
            for idx in sample_idx_list[k]
        ]
        y_n = [
            int(grouped_train[k][idx][1]) 
            for k in range(K)
            for idx in sample_idx_list[k]
        ]
        client_ds = tf.data.Dataset.from_tensor_slices((x_n, y_n))
        save_client_datasets(client_ds, uname, base_dir)


        label_dist = np.zeros(K)
        for label in y_n:
            label_dist[label] += 1
        print(num_samples_train[n], len(y_n))
        label_dist = label_dist / len(y_n)
        train_data[uname]["distribution"] = list(label_dist)
        train_data[uname]["num_samples"] = len(y_n)

    print("Training partition completed.")

    uname = "test"
    test_label_n = choice(K, num_samples_test, p=distribution_test[0])
    freq = np.zeros(K)
    for k in test_label_n:
        freq[k] += 1
    sample_idx_list = [
        choice(len(grouped_test[k]), min(int(freq[k]), 100), replace=False) 
        for k in range(K)
    ]         
    x_n = [
        tf.squeeze(grouped_test[k][idx][0]) 
        for k in range(K)
        for idx in sample_idx_list[k]
    ]
    y_n = [
        int(grouped_test[k][idx][1]) 
        for k in range(K)
        for idx in sample_idx_list[k]
    ]
    client_ds = tf.data.Dataset.from_tensor_slices((x_n, y_n))
    save_client_datasets(client_ds, uname, base_dir)
    print("Model is saved.")

    label_dist = np.zeros(K)
    for label in y_n:
        label_dist[label] += 1
    print(num_samples_test, len(y_n))
    label_dist = label_dist / num_samples_test
    test_data[uname]["distribution"] = list(label_dist)
    test_data[uname]["num_samples"] = len(y_n)

    
    print("Testing partition completed.")
    

    with open(base_dir+"train_data.pck", "wb") as f:
            pickle.dump(train_data, f)
    with open(base_dir+"test_data.pck", "wb") as f:
            pickle.dump(test_data, f)

    train_dist = np.array(
        [train_data[uname]["distribution"] for uname in uname_list]
    )
    train_dist = sum(train_dist) / N

    test_dist = np.array(test_data["test"]["distribution"])

    show_hist(train_dist, test_dist)
    print(f"overall KL is {KL(test_dist, train_dist)}")
    return train_data, test_data


dataset_dir = "data/tiny-imagenet-200/train"
# restructure_tiny_imagenet(dataset_dir)
# print("Restruction completed")

batch_size = 1
img_size = (64, 64)  # Resize images if needed

dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir, image_size=img_size, batch_size=batch_size, shuffle=False
)

dataset_size = sum(1 for _ in dataset)
print("Dataset size:", dataset_size)


print("Loading completed")

for v in ["control"]:
    train_data, test_data = noniid_partition(v, dataset, alpha_train=0.5, alpha_test=0.01)