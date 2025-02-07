import os

# Only the TensorFlow backend supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pathlib
import numpy as np
import keras
from keras import layers
import csv    
import tensorflow.data as tf_data
from models import get_MLPmodel

train_path = "data/Sent140/trainingandtestdata/training.1600000.processed.noemoticon.csv"
test_path = "data/Sent140/trainingandtestdata/testdata.manual.2009.06.14.csv"
glove_path = "glove.twitter.27B/glove.twitter.27B.200d.txt"
embedding_dim = 200

X_train = []
Y_train = []
X_test = []
Y_test = []

with open(train_path, "r", encoding='latin-1') as data:
    reader = csv.reader(data)
    next(reader)
    for line in reader:
        row = np.array([v for v in line])
        feature = np.array(row[-1]) # Get only text
        label = int(row[0])
        if label == 2: # drop label 2 (neutral)
            continue
        label = int(label/4) # transform 0, 4 to 0, 1
        X_train.append(feature)
        Y_train.append(label)

with open(test_path, "r", encoding='latin-1') as data:
    reader = csv.reader(data)
    next(reader)
    for line in reader:
        row = np.array([v for v in line])
        feature = np.array(row[-1])
        label = int(row[0])
        if label == 2: # drop label 2 (neutral)
            continue
        label = int(label/4) # transform 0, 4 to 0, 1
        X_test.append(feature)
        Y_test.append(label)

print(X_train[0], Y_train[0])
print(X_test[0], Y_test[0])
print(np.unique(np.array(Y_train)), np.unique(np.array(Y_test)))
vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf_data.Dataset.from_tensor_slices(X_train).batch(32)
vectorizer.adapt(text_ds)
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))
test = ["the", "cat", "sat", "on", "the", "mat"]
print([word_index[w] for w in test])

embeddings_index = {}
with open(glove_path) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")

num_tokens = len(voc) + 2
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print(f"Converted {hits} words ({misses} misses)")

model = get_MLPmodel(num_tokens, embedding_dim, embedding_matrix)
model.summary()

X_train = vectorizer(np.array([[v] for v in X_train])).numpy()
Y_train = np.array(Y_train)
X_test = vectorizer(np.array([[v] for v in X_test])).numpy()
Y_test = np.array(Y_test)
# print(model(X_train[0]), Y_train[0])
model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=2)
test_scores = model.evaluate(X_test, Y_test, verbose=0)
print(test_scores)


