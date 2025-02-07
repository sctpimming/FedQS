import keras
from keras import layers
import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np
import mobilenet_v2
from pgd import PerturbedGradientDescent
from keras.optimizers import Optimizer
import keras.backend as K



class ProximalSGD(Optimizer):
    def __init__(self, learning_rate=0.01, l1_lambda=0.01, **kwargs):
        super(ProximalSGD, self).__init__(name="ProximalSGD", **kwargs)
        self.learning_rate = learning_rate  # Learning rate as a float
        self.l1_lambda = l1_lambda  # Regularization parameter for L1 regularization

    def _create_slots(self, var_list):
        # No slots needed since momentum is removed
        pass

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # Apply the standard SGD update
        var.assign_sub(self.learning_rate * grad)  # Directly use the learning rate

        # Apply the proximal operator (L1 regularization)
        var.assign(self.proximal_operator(var))

    def proximal_operator(self, var):
        """Apply the proximal operator for L1 regularization."""
        return tf.sign(var) * tf.maximum(tf.abs(var) - self.l1_lambda, 0)

    def set_learning_rate(self, new_learning_rate):
        """Update the learning rate."""
        self.learning_rate = new_learning_rate

    def get_config(self):
        config = super(ProximalSGD, self).get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "l1_lambda": self.l1_lambda,
        })
        return config


# Get a "l2 norm of gradients" tensor
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

def get_LeNet(K=100, input_shape=(32, 32, 3), config="None", weight=None):
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
            optimizer=keras.optimizers.SGD(weight_decay=0.00004),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
    # model.metrics_names.append("gradient_norm")
    # model.metrics_tensors.append(get_gradient_norm(model))

    return model


def get_MobileNet(K=1203, input_shape=(224, 224, 3), config="None", weight=None):
    inputs = tf.keras.Input(shape=input_shape)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        # pooling="max",
    )
    # base_model.summary()
    # layers_list = [l for l in base_model.layers]
    # Mobile_GN = insert_layer_nonseq(base_model, '.*BN.*', GN_factory)
    # Mobile_GN.save('temp.h5')
    # Mobile_GN = keras.models.load_model('temp.h5')
    # Mobile_GN.summary()
    x = base_model(inputs)
    fc2 = layers.Dense(K, activation="softmax")
    outputs = fc2(x)
    # print(x.shape)
    # x = tf.keras.layers.AveragePooling2D(7)(x)
    # x = tf.keras.layers.Conv2D(K, (1, 1), padding='same')(x)
    # x = tf.keras.layers.Reshape((K,), name='output')(x)
    # outputs = tf.keras.layers.Activation('softmax', name='final_activation')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    #model.summary()
    # for name in layers_name:
    #     print(name, base_model.get_layer(name))
    # print(layers_name)
    if config == "prox":
        # optimizer = tfa.optimizers.ProximalAdagrad(learning_rate=0.01, l2_regularization_strength=0.1)
        optimizer = keras.optimizers.SGD(weight_decay=0.00004)
        model.compile(optimizer=optimizer,
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    else:
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(),
            optimizer=keras.optimizers.SGD(weight_decay=0.00004),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    return model

def get_ResNet50(weight_config=None, K=200):
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
    model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(weight_decay=1e-5, momentum=0.9),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# model = get_ChoNet()
