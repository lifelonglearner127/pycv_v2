# USAGE
# python extract_features.py --dataset ~/datasets/ --output features.hdf5

import argparse
import random
import os
import numpy as np
import progressbar
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from pycv.datasets import HDF5DatasetWriter


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
                help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
                help="size of feature extraction buffer")
args = vars(ap.parse_args())

batch_size = args["batch_size"]

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
random.shuffle(image_paths)

labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter(
    (len(image_paths), 512 * 7 * 7), args["output"],
    data_key="features", buf_size=args["buffer_size"]
)
dataset.store_class_labels(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(
    maxval=len(image_paths), widgets=widgets
).start()

for i in np.arange(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    batch_labels = labels[i:i + batch_size]
    batch_images = []

    for (j, image_path) in enumerate(batch_paths):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)

        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batch_images.append(image)

    batch_images = np.vstack(batch_images)
    features = model.predict(batch_images, batch_size=batch_size)

    features = features.reshape((features.shape[0], 512 * 7 * 7))

    dataset.add(features, batch_labels)
    pbar.update(i)

dataset.close()
pbar.finish()
