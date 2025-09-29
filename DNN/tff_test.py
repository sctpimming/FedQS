import collections
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import InputLayer, Layer, BatchNormalization, GroupNormalization
import tensorflow_federated as tff
import keras
from tensorflow.keras import backend as K
from tqdm import tqdm
import wandb
import json
import models

import os


print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow Federated version: {tff.__version__}")

# --- 1. CONFIGURATION AND HYPERPARAMETERS ---
NUM_CLIENTS = 9275
NUM_PARTICIPANTS = 10
NUM_ROUNDS = 10
BATCH_SIZE = 16
NUM_EPOCHS = 5
IS_GROUP_NORM = False
LEARNING_RATE = 0.01
SUBSAMPLE_SIZE = 3200

# tff.backends.native.set_sync_local_cpp_execution_context(max_concurrent_computation_calls=1)

# --- Data Loading and Preprocessing ---
TARGET_IMAGE_CHANNELS = 3
TARGET_IMAGE_SIZE = (299, 299)

# Define the image and label specifications
image_spec = tf.TensorSpec(shape=(None, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], TARGET_IMAGE_CHANNELS), dtype=tf.float32)
label_spec = tf.TensorSpec(shape=(None,), dtype=tf.int64)

# Combine them into a tuple that matches the dataset output
batched_input_spec = (image_spec, label_spec)

tff.backends.native.set_sync_local_cpp_execution_context(
    max_concurrent_computation_calls=10
)

def get_metadata():
    dataset = "inat"
    train_path = f"./{dataset}_metadata/train_metadata.json"
    test_path = f"./{dataset}_metadata/test_metadata.json"
    with open(train_path, "rb") as f:
        train_metadata = json.load(f)
    with open(test_path, "rb") as f:
        test_metadata = json.load(f)
    
    return train_metadata, test_metadata

def load_tfrecord_dataset(
    tfrecord_file_paths: str, # Use a single path for TFF
    is_training: bool = True,
    compression_type: str = None
):
    """
    Loads and preprocesses a single TFRecord file for a TFF client.
    """
    if not isinstance(tfrecord_file_paths, str):
        raise TypeError("tfrecord_file_paths should be a single string for TFF.")

    if not os.path.exists(tfrecord_file_paths):
        raise FileNotFoundError(f"TFRecord file not found: {tfrecord_file_paths}")

    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'original_inat_label': tf.io.FixedLenFeature([], tf.int64),
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'user_id': tf.io.FixedLenFeature([], tf.string),
        'inat_2017_category_id': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_and_preprocess_example(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)

        image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=TARGET_IMAGE_CHANNELS)
        image = tf.cast(image, tf.float32)

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
            image = tf.reshape(image, (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], TARGET_IMAGE_CHANNELS))
            # image = tf.image.random_flip_left_right(image) # Optional: uncomment for more augmentation
        else:
            image = tf.image.resize(image, TARGET_IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
            image = tf.reshape(image, (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], TARGET_IMAGE_CHANNELS))


        # --- NORMALIZATION STEP ---
        image = mobilenet_v2_preprocess_input(image)

        label = parsed_features['label']
        label = tf.ensure_shape(label, [])

        return image, label

    # Create the dataset
    dataset = tf.data.TFRecordDataset(
        tfrecord_file_paths,
        compression_type=compression_type,
        num_parallel_reads=tf.data.AUTOTUNE
    )

    # Apply the parsing and preprocessing map
    dataset = dataset.map(_parse_and_preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        # TFF's training loop will handle shuffling and batching
        dataset = dataset.shuffle(buffer_size=64)
    return dataset.prefetch(tf.data.AUTOTUNE)


train_metadata, test_metadata = get_metadata()
user_name = list(train_metadata.keys())
nsample_train = [train_metadata[uname]["num_examples"] for uname in user_name]
szfrac_train = np.array(
    [nsample_train[n] / sum(nsample_train) for n in range(NUM_CLIENTS)]
)

# Load client datasets from your .tfrecord files
# This assumes your files are named "user_0.tfrecord", "user_1.tfrecord", etc.
client_file_paths = [f"./inat_train/user_{i}.tfrecord" for i in range(NUM_CLIENTS)]

# Load the centralized test set
centralized_test_data = load_tfrecord_dataset("./inat_test/inat_test_set.tfrecord", is_training=False)

subsampled_test_data = centralized_test_data.take(SUBSAMPLE_SIZE).batch(BATCH_SIZE)

# --- 2. MODEL DEFINITION ---
def get_MobileNet(K=1203, input_shape=(299, 299, 3), IsGN=False, learning_rate=0.01):
    # This is the base model from Keras, which uses BatchNormalization
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    
    if IsGN:
        base_model = models.replace_batchnorm_with_groupnorm(base_model)
    
    x = base_model.output
    
    fc2 = layers.Dense(K, activation="softmax")
    outputs = fc2(x)
    
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    return model

# TFF model wrapper
def tff_model_fn():
    keras_model = get_MobileNet(IsGN=IS_GROUP_NORM, learning_rate=LEARNING_RATE)
    
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    return tff.learning.models.from_keras_model(
        keras_model,
        loss=loss_fn,
        input_spec=batched_input_spec,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# --- 3. TFF ALGORITHM SETUP ---

# TFF process for FedAvg
federated_algorithm = tff.learning.algorithms.build_weighted_fed_avg(
    tff_model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.01),
    server_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=1.0),
)
state = federated_algorithm.initialize()

eval_process = tff.learning.algorithms.build_fed_eval(tff_model_fn)
eval_state = eval_process.initialize()

# --- 4. SIMULATION LOOP ---

# wandb.init(project="TFF-FedAVG", config={
#     "num_rounds": NUM_ROUNDS,
#     "num_clients": NUM_CLIENTS,
#     "num_local_epochs": NUM_EPOCHS,
#     "batch_size": BATCH_SIZE,
# })

client_weights = szfrac_train
client_weights = client_weights/sum(client_weights)

for round_num in tqdm(range(NUM_ROUNDS), desc="Federated Learning Rounds"):
    # Select a random subset of clients for this round
    client_indices = np.random.choice(NUM_CLIENTS, size=NUM_PARTICIPANTS, replace=False, p=client_weights)
    selected_client_file_paths = [client_file_paths[i] for i in client_indices]
    # Dynamically create the dataset for the selected client(s)
    selected_client_datasets = [
        load_tfrecord_dataset(f, is_training=True)
        .repeat(NUM_EPOCHS)
        .batch(BATCH_SIZE)
        for f in selected_client_file_paths
    ]
    print("Data loading completed")
    # Run a federated step
    state, metrics = federated_algorithm.next(state, selected_client_datasets)
    
    print(f"Round {round_num + 1}/{NUM_ROUNDS}")
    metrics_train = metrics['client_work']['train']
    print(f"Loss: {metrics_train['loss']:.4f}, Accuracy: {metrics_train['sparse_categorical_accuracy']:.4f}")

    if (round_num) % 5 == 0:
        model_weights = federated_algorithm.get_model_weights(state)
        eval_state = eval_process.set_model_weights(eval_state, model_weights)
        eval_output = eval_process.next(eval_state, [subsampled_test_data])
        
        eval_metrics = eval_output.metrics['client_work']['eval']['current_round_metrics']
        print(f"Test Loss: {eval_metrics['loss']:.4f}, Test Accuracy: {eval_metrics['sparse_categorical_accuracy']:.4f}")

    # wandb.log({
    #     "average_train_loss": metrics_train['loss'],
    #     "average_train_acc": metrics_train['sparse_categorical_accuracy'],
    #     "global_test_loss": eval_metrics['loss'],
    #     "global_test_accuracy": eval_metrics['sparse_categorical_accuracy']
    # }, step=round_num)
      
