import argparse
from imutils import paths
from pycv.preprocessor import ResizePreProcessor
from pycv.datasets import DatasetLoader
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))

resize_preprocessor = ResizePreProcessor(32, 32)
dataset_loader = DatasetLoader(preprocessors=[resize_preprocessor])
(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
le = le.fit_transform(labels)

(train_X, test_X, train_y, test_y) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

for r in (None, "l1", "l2"):
    print(f"[INFO] training model with `{r}` penalty")
    model = SGDClassifier(loss="log", penalty=r, max_iter=10,
                          learning_rate="constant", tol=1e-3, eta0=0.01,
                          random_state=12)
    model.fit(train_X, train_y)

    acc = model.score(test_X, test_y)
    print(f"[INFO] `{r}` penalty accuracy: {acc * 100:.2f}%")
