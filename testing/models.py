import keras
from keras import layers


def get_MLPmodel(num_tokens, embedding_dim, embedding_matrix):
    embedding_layer = layers.Embedding(num_tokens, embedding_dim, trainable=False) # Freeze pretrained layer
    embedding_layer.build((1, ))
    embedding_layer.set_weights([embedding_matrix])
    model = keras.Sequential(
        [
            keras.Input(shape=(None,), dtype="int32"),
            embedding_layer,
            layers.GlobalAveragePooling1D(),
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


def get_CNNmodel(img_shape = (32, 32, 3)):
    model = keras.Sequential()
    model.add(layers.Conv2D(6, (5, 5), input_shape=img_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(16, (5, 5)))
    # model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(120,))
    model.add(layers.Dense(84))
    model.add(layers.Dense(10, activation="softmax"))
    # model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(weight_decay=0.0005),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def get_CNNmodel_cho():
    model = keras.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation="relu", input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation="relu"))
    # model.summary()

    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(84, activation="relu"))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    # model.summary()
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.SGD(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model




