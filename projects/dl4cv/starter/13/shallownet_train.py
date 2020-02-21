# USAGE
# python shallownet_train.py --dataset ~/datasets/animals --model model.hdf5

import argparse
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from pycv.datasets import DatasetLoader
from pycv.networks import ShallowNet
from pycv.preprocessors import ImageToArrayPreProcessor
from pycv.preprocessors import ResizePreProcessor


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="Path to output model")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
rp = ResizePreProcessor(width=32, height=32)
iap = ImageToArrayPreProcessor()

dataset_loader = DatasetLoader(preprocessors=[rp, iap])
(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(train_X, test_X, train_y, test_y) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)
train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print("[INFO] compiling model...")
sgd = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)

print("[INFO] training network...")
H = model.fit(
    train_X, train_y, validation_data=(test_X, test_y),
    batch_size=32, epochs=100, verbose=1
)

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=32)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=["cat", "dog", "pandas"]
    )
)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
