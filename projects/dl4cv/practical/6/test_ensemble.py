import argparse
import glob
import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
                help="path to models directory")
args = vars(ap.parse_args())


(test_X, test_y) = cifar10.load_data()[1]
test_X = test_X.astype("float") / 255.0

label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

lb = LabelBinarizer()
test_y = lb.fit_transform(test_y)

model_paths = os.path.sep.join([args["models"], "*.model"])
model_paths = list(glob.glob(model_paths))
models = []

for (i, model_path) in enumerate(model_paths):
    print("[INFO] loading model {}/{}".format(i + 1, len(model_paths)))
    models.append(load_model(model_path))

print("[INFO] evaluating ensemble...")
predictions = []

for model in models:
    predictions.append(model.predict(test_X, batch_size=64))

predictions = np.average(predictions, axis=0)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=label_names
    )
)
