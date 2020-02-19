# USAGE
# python shallownet_load.py --dataset ~/datasets/animals --model model.hdf5

import argparse
import cv2
import numpy as np
from imutils import paths
from tensorflow.keras.models import load_model
from pycv.datasets import DatasetLoader
from pycv.preprocessors import ImageToArrayPreProcessor
from pycv.preprocessors import ResizePreProcessor


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="Path to pre-trained model")
args = vars(ap.parse_args())

class_labels = ["cat", "dog", "panda"]

print("[INFO] sampling images...")
image_paths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(image_paths), size=(10, ))
image_paths = image_paths[idxs]

rp = ResizePreProcessor(width=32, height=32)
iap = ImageToArrayPreProcessor()

dataset_loader = DatasetLoader(preprocessors=[rp, iap])
(data, labels) = dataset_loader.load(image_paths)
data = data.astype("float") / 255.0

print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
predictions = model.predict(data, batch_size=32).argmax(axis=1)

for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    cv2.putText(image, f"{class_labels[predictions[i]]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
