import tensorflow as tf
from keras import layers, Model
from keras.layers import Layer, Input, InputLayer, GroupNormalization, BatchNormalization
import keras
# import tensorflow_addons as tfa
import numpy as np
import mobilenet_v2
from pgd import PerturbedGradientDescent
from keras.optimizers import Optimizer
import keras.backend as K
from collections import deque
import gc

import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, GroupNormalization, Input
from tensorflow.keras import Model, initializers
from tensorflow.keras.backend import int_shape

def replace_batchnorm_with_groupnorm(original_model, groups_sz=None, group_num=None):
    new_layers = {}
    input_tensor = Input(tensor=original_model.input, name=original_model.input.name)
    tensor_map = {original_model.input.ref(): input_tensor}
    
    for layer in original_model.layers:
        if isinstance(layer, InputLayer):
            continue

        input_tensors = []
        if isinstance(layer.input, (list, tuple)):
            for t in layer.input:
                input_tensors.append(tensor_map[t.ref()])
        else:
            input_tensors = tensor_map[layer.input.ref()]

        if isinstance(layer, BatchNormalization):
            num_channels = int_shape(input_tensors)[-1]
            
            if groups_sz is not None:
                current_groups = num_channels // groups_sz
            elif group_num is not None:
                current_groups = group_num
            else:
                # Default to 32 groups if no size is specified
                current_groups = num_channels
            
            is_last_bn_in_block = 'add' in original_model.layers[original_model.layers.index(layer) + 1].name

            if is_last_bn_in_block:
                gamma_initializer = initializers.Zeros() 
            else:
                gamma_initializer = initializers.Ones()

            new_layer = GroupNormalization(
                groups=current_groups,
                name=layer.name,
                epsilon=layer.epsilon,
                gamma_initializer=gamma_initializer
            )
            new_output_tensor = new_layer(input_tensors)

        else:
            config = layer.get_config()
            new_layer = layer.__class__(**config)
            
            # Call the new layer on the input tensor to build it
            new_output_tensor = new_layer(input_tensors)
            
            # Now that the layer is built, we can safely set the weights
            if layer.get_weights():
                new_layer.set_weights(layer.get_weights())

        tensor_map[layer.output.ref()] = new_output_tensor
        new_layers[layer.name] = new_layer

    new_model = Model(inputs=input_tensor, outputs=new_output_tensor)
    return new_model

def check_replace_batchnorm(original_model):
    new_layers = {}
    input_tensor = Input(tensor=original_model.input, name=original_model.input.name)
    tensor_map = {original_model.input.ref(): input_tensor}

    for layer in original_model.layers:
        if isinstance(layer, InputLayer):
            continue

        input_tensors = []
        if isinstance(layer.input, (list, tuple)):
            for t in layer.input:
                input_tensors.append(tensor_map[t.ref()])
        else:
            input_tensors = tensor_map[layer.input.ref()]

        if isinstance(layer, BatchNormalization):
            # For sanity check, replace with a NEW BatchNormalization layer
            new_layer = BatchNormalization(
                name=layer.name,
                epsilon=layer.epsilon,
                axis=layer.axis
            )
            new_output_tensor = new_layer(input_tensors)

        else:
            config = layer.get_config()
            new_layer = layer.__class__(**config)
            new_output_tensor = new_layer(input_tensors)
            if layer.get_weights():
                new_layer.set_weights(layer.get_weights())

        tensor_map[layer.output.ref()] = new_output_tensor
        new_layers[layer.name] = new_layer

    new_model = Model(inputs=input_tensor, outputs=new_output_tensor)
    return new_model


def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

def get_MLPmodel(num_tokens, embedding_dim, embedding_matrix):
    embedding_layer = layers.Embedding(num_tokens, embedding_dim, trainable=False) # Freeze pretrained layer
    embedding_layer.build((1, ))
    embedding_layer.set_weights([embedding_matrix])
    model = keras.Sequential(
        [
            keras.Input(shape=(None,), dtype="int32"),
            embedding_layer,
            layers.Dense(128),
            layers.Dense(86),
            layers.Dense(30),
            layers.Dense(2, activation="softmax")
        ]
    )
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def get_ChoNet(img_shape = (32, 32, 3), config="None"):
    model = keras.Sequential()
    model.add(layers.Conv2D(6, (5, 5), input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(84, activation ="relu"))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def get_CNNmodel(img_shape = (32, 32, 3), config="None"):
    model = keras.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation="relu", input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(weight_decay=0.0005),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def get_LeNet(K=100, input_shape=(32, 32, 3), config="None", weight=None, learning_rate=0.01):
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(384, activation="relu"))
    model.add(layers.Dense(192, activation="relu"))
    model.add(layers.Dense(K, activation="softmax"))
    # model.summary()
    if config == "prox":
        prox_sgd = tfa.optimizers.ProximalAdagrad(learning_rate=0.01, l2_regularization_strength=0.001)
        model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer= prox_sgd,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    else:
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.SGD(weight_decay=0.00004, learning_rate=learning_rate),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    # model.metrics_names.append("gradient_norm")
    # model.metrics_tensors.append(get_gradient_norm(model))

    return model


def get_MobileNet(K=1203, input_shape=(224, 224, 3), IsGN=False, learning_rate=0.01, group_sz=None, group_num=None):
    # Ensure the input shape is correct for MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    
    # This is the correct logic for Group Normalization
    if IsGN:
        base_model = replace_batchnorm_with_groupnorm(base_model, groups_sz=group_sz, group_num=group_num)

    # Get the output of the base model
    x = base_model.output
    
    # Add the final classification layer
    fc2 = layers.Dense(K, activation="softmax")
    outputs = fc2(x)
    
    # Build the complete model with the base model's input and new output
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    
    
    # Compile the model with the specified optimizer and metrics
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(weight_decay=0.0005, learning_rate=learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def get_ResNet50(weight_config=None, K=200, rseed=0):
    tf.random.set_seed(int(rseed))
    inputs = tf.keras.Input(shape=(64, 64, 3))

    base_model = keras.applications.ResNet50(
        input_shape=(64, 64, 3),
        include_top=False,
        pooling="avg",
        weights=weight_config,
    )
    

    fc1 = layers.Dense(2048, activation="relu")
    fc2 = layers.Dense(256)
    predict = layers.Dense(K, activation="softmax")
    x = base_model(inputs)
    x = fc1(x)
    x = fc2(x)
    outputs = predict(x)

    model = tf.keras.Model(inputs, outputs)
    # model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(weight_decay=1e-5, momentum=0.9),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model



# model = get_ChoNet()
