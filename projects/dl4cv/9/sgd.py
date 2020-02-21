# USAGE
# python sgd.py

import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
                help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
                help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="size of SGD mini-batches")
args = vars(ap.parse_args())

(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=42)
y = y.reshape((y.shape[0], 1))
X = np.c_[X, np.ones((X.shape[0]))]

(train_X, test_X, train_y, test_y) = train_test_split(X, y, test_size=0.5,
                                                      random_state=42)

print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in np.arange(0, args["epochs"]):
    epoch_loss = []

    for (batch_X, batch_y) in next_batch(train_X, train_y, args["batch_size"]):
        preds = sigmoid_activation(batch_X.dot(W))

        error = preds - batch_y
        epoch_loss.append(np.sum(error ** 2))

        d = error * sigmoid_deriv(preds)
        gradient = batch_X.T.dot(d)
        W += -args["alpha"] * gradient

    loss = np.average(epoch_loss)
    losses.append(loss)
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))

print("[INFO] evaluating...")
preds = predict(test_X, W)
print(classification_report(test_y, preds))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(test_X[:, 0], test_X[:, 1], marker="o", c=test_y[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
