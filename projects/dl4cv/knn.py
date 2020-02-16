import argparse

from imutils import paths
from pycv.preprocessor import ResizePreProcessor
from pycv.datasets import DatasetLoader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to input datasets")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN (-1 use all available cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

resize_preprocessor = ResizePreProcessor(32, 32)
dataset_loader = DatasetLoader(preprocessors=[resize_preprocessor])

(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print(f"[INFO] features matrix: {data.nbytes / (1024 * 1024.0):.1f}MB")

le = LabelEncoder()
labels = le.fit_transform(labels)

(train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                      test_size=0.25,
                                                      random_state=42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(train_x, train_y)

print(classification_report(test_y, model.predict(test_x),
                            target_names=le.classes_))
