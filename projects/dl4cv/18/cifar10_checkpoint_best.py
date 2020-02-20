# USAGE
# python cifar10_checkpoint_best.py --weights weights/best/best_weights.hdf5

import argparse
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from pycv.networks import MiniVGGNet


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
                help="path to best model weights file")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.fit_transform(test_y)

print("[INFO] compiling model...")
sgd = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    args["weights"], monitor="val_loss", save_best_only=True, verbose=1
)
callbacks = [checkpoint]

print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              batch_size=64, epochs=40, callbacks=callbacks, verbose=1)
