# USAGE
# python minivggnet_flowers.py --dataset ~/datasets/flowers
# --output without.png

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from pycv.datasets import DatasetLoader
from pycv.networks import MiniVGGNet
from pycv.preprocessors import AspectAwarePreProcessor
from pycv.preprocessors import ImageToArrayPreProcessor


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
class_names = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aap = AspectAwarePreProcessor(64, 64)
iap = ImageToArrayPreProcessor()
dataset_loader = DatasetLoader(preprocessors=[aap, iap])
(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(train_X, test_X, train_y, test_y) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print("[INFO] compiling model...")
sgd = SGD(lr=0.05)
model = MiniVGGNet.build(
    width=64, height=64, depth=3, classes=len(class_names)
)
model.compile(
    loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
)

print("[INFO] training network...")
H = model.fit(train_X, train_y, validation_data=(test_X, test_y),
              batch_size=32, epochs=100, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(test_X, batch_size=32)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=class_names
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
plt.show()
