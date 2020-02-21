# USAGE
# python lenet_mnist.py

import matplotlib.pyplot as plt
import numpy as np
from pycv.networks import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD


print("[INFO] accessing MNIST...")
((train_X, train_y), (test_X, test_y)) = mnist.load_data()

if K.image_data_format() == "channels_first":
    train_X = train_X.reshape(train_X.shape[0], 1, 28, 28)
    test_X = test_X.reshape(test_X.shape[0], 1, 28, 28)
else:
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

train_X = train_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

print("[INFO] compiling model...")
sgd = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)

print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=128)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=[str(x) for x in lb.classes_]
    )
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
