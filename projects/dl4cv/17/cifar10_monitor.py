import argparse
import os
import matplotlib
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from pycv.callbacks import TrainingMonitor
from pycv.networks import MiniVGGNet


matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to the output directory")
args = vars(ap.parse_args())

print("[INFO process ID: {}".format(os.getpid()))

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
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)

fig_path = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
json_path = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(fig_path, json_path=json_path)]
model.fit(train_X, train_y, validation_data=(test_X, test_y),
          batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
