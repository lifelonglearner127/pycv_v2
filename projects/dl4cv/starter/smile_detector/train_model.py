# USAGE
# python train_model.py --dataset ~/datasets/smiles --model lenet.hdf5

import argparse
import os
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from pycv.networks import LeNet


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset of faces")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

data = []
labels = []

image_paths = list(paths.list_images(args["dataset"]))
for (i, image_path) in enumerate(sorted(image_paths)):
    if i > 0 and i % 100 == 0:
        print(f"[INFO] processed {i} / {len(image_paths)} images...")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals
(train_X, test_X, train_y, test_y) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              class_weight=class_weight, batch_size=64, epochs=15, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=64)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=le.classes_
    )
)

print("[INFO] serializing network...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["accuracy"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
