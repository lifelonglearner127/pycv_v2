import os
import cv2
import h5py
import numpy as np

from tensorflow.keras.utils import to_categorical


class DatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        data = []
        labels = []

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            for p in self.preprocessors:
                image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(image_paths)}")

        return (np.array(data), np.array(labels))


class HDF5DatasetWriter:

    def __init__(self, dims, output_path, data_key="images", buf_size=1000):
        """HDF5DatasetWriter Constructor

        Key Arguments:
        dims: dimension or shape of the data we will be storing in the dataset.
        output_path: path to where our output HDF5 file will be stored
        data_key: name of the dataset
        buf_size: size of our in-memory buffer
        """
        if os.path.exists(output_path):
            raise ValueError(
                "The supplied `outputPath` already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", output_path
            )

        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0], ),
                                             dtype="int")
        self.buf_size = buf_size
        self.buffer = {
            "data": [],
            "labels": []
        }
        self.idx = 0

    def add(self, rows, labels):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.buf_size:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {
            "data": [],
            "labels": []
        }

    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset(
            "label_names", (len(class_labels), ), dtype=dt
        )
        label_set[:] = class_labels

    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()


class HDF5DatasetGenerator:

    def __init__(self, db_path, batch_size, preprocessors=None,
                 aug=None, binarize=True, classes=2):
        """HDF5DatasetGenerator Constructor

        Key Arguments:
        db_path: path to out hdf5 dataset that stores images
                 and corresponding class labels
        batch_size: the size of mini-batches to yield when training network
        preprocessors: list of image preprocessors
        aug: Keras ImageDataGenerator to apply data agumentation directly
        binarize: Indicates whether or not binarization of class labels
                  needs to take place
        classes: The number of unique class labels
        """
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        self.db = h5py.File(db_path, "r")
        self.num_images = self.db["labels"].shape[0]

    def generator(self, passes=np.inf):
        epochs = 0

        while epochs < passes:
            for i in np.arange(0, self.num_images, self.batch_size):
                images = self.db["images"][i:i + self.batch_size]
                labels = self.db["labels"][i:i + self.batch_size]

                if self.binarize:
                    labels = to_categorical(labels, self.classes)

                if self.preprocessors is not None:
                    proc_images = []

                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)

                    images = np.array(proc_images)

                if self.aug is not None:
                    (images, labels) = next(
                        self.aug.flow(images, labels,
                                      batch_size=self.batch_size)
                    )

                yield (images, labels)

            epochs += 1

    def close(self):
        self.db.close()
