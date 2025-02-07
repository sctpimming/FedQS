import numpy as np
from util.misc import softmax, mod_softmax
from sklearn.metrics import f1_score
from math import log


def crossentropy(w, data, B, K, C, M, mu):  # TODO: calculate sz from data
    sz = len(data["y"])
    batch_idx = np.random.choice(sz, B, replace=False)
    sumloss = 0
    for i in batch_idx:
        pred = [np.dot(w[k], data["x"][i]) for k in range(K)]
        pclass = mod_softmax(pred, C, M)
        for k in range(K):
            if data["y"][i] == k:
                sumloss = sumloss - log(pclass[k])
    return (sumloss / B) + mu * (np.linalg.norm(w) ** 2)


def accuracy(w, data, B, K, C, M, perclass=False):
    sz = len(data["y"])
    batch_idx = np.random.choice(sz, B, replace=False)
    class_cnt = np.zeros(K)
    class_acc = np.zeros(K)
    confmat = np.zeros((K, K))
    for i in batch_idx:
        pred = [np.dot(w[k], data["x"][i]) for k in range(K)]
        pclass = mod_softmax(pred, C, M)

        predicted_class = np.argmax(pclass)
        actual_class = int(data["y"][i])
        class_cnt[actual_class] += 1
        confmat[predicted_class][actual_class] += 1

    if perclass == True:
        class_correct = np.diagonal(confmat)
        for k in range(K):
            if class_cnt[k] > 0:
                class_acc[k] = class_correct[k] / class_cnt[k]
        return class_acc
    return np.sum(np.diagonal(confmat) / B)

def macro_f1(w, data, B, K, C, M):
    sz = len(data["y"])
    batch_idx = np.random.choice(sz, B, replace=False)
    y_pred = np.zeros(B)
    y_true = np.zeros(B)
    idx = 0
    confmat = np.zeros((K, K))
    for i in batch_idx:
        pred = [np.dot(w[k], data["x"][i]) for k in range(K)]
        pclass = mod_softmax(pred, C, M)

        y_pred[idx] = np.argmax(pclass)
        y_true[idx]= int(data["y"][i])
        idx += 1
    return f1_score(y_true, y_pred, average="macro")
    