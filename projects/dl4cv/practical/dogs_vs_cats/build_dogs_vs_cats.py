import json
import os

import cv2
import numpy as np
import progressbar

import config

from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from pycv.preprocessors import AspectAwarePreProcessor
from pycv.datasets import HDF5DatasetWriter


train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [p.split(os.path.sep)[-1].split(".")[0] for p in train_paths]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

split = train_test_split(train_paths, train_labels,
                         test_size=config.NUM_TEST_IMAGES,
                         stratify=train_labels, random_state=42)
(train_paths, test_paths, train_labels, test_labels) = split

split = train_test_split(train_paths, train_labels,
                         test_size=config.NUM_VAL_IMAGES,
                         stratify=train_labels, random_state=42)
(train_paths, val_paths, train_labels, val_labels) = split

datasets = [
    ("train", train_paths, train_labels, config.TRAIN_HDF5),
    ("val", val_paths, val_labels, config.VAL_HDF5),
    ("test", test_paths, test_labels, config.TEST_HDF5)
]

aap = AspectAwarePreProcessor(256, 256)
(R, G, B) = ([], [], [])

for (dataset_type, image_paths, labels, output_path) in datasets:
    print(f"[INFO] building {output_path}...")
    writer = HDF5DatasetWriter((len(image_paths), 256, 256, 3), output_path)

    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(
        maxval=len(image_paths), widgets=widgets).start()

    for (i, (image_path, label)) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(image_path)
        image = aap.preprocess(image)

        if dataset_type == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
