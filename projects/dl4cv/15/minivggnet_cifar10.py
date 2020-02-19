import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from pycv.networks import MiniVGGNet


matplotlib.use("Agg")


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model...")
sgd = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=sgd,
              metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              batch_size=64, epochs=40, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=64)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=label_names
    )
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
