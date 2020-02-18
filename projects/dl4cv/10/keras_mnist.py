# USAGE
# python keras_mnist.py --output output/keras_mnist.png

import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] accessing MNIST...")
((train_X, train_y), (test_X, test_y)) = mnist.load_data()

train_X = train_X.reshape((train_X.shape[0], 28 * 28 * 1))
test_X = test_X.reshape((test_X.shape[0], 28 * 28 * 1))

train_X = train_X.astype("float32") / 255.0
test_X = test_X.astype("float32") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

model = Sequential()
model.add(Dense(256, input_shape=(784, ), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))

print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])

H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              epochs=100, batch_size=128)

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=128)
print(
    classification_report(
        test_y.argmax(axis=1), predictions.argmax(axis=1),
        target_names=[str(x) for x in lb.classes_]
    )
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
