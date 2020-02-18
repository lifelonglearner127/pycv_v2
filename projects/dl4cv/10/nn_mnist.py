# USAGE
# python nn_mnist.py

from pycv.networks import NeuralNetwork
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data - data.min()) / (data.max() - data.min())
print(f"[INFO] samples: {data.shape[0]}, dim: {data.shape[1]}")

(train_X, test_X, train_y, test_y) = train_test_split(
    data, digits.target, test_size=0.25
)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

print("[INFO] training network...")
nn = NeuralNetwork([train_X.shape[1], 32, 16, 10])
print(f"[INFO] {nn}")
nn.fit(train_X, train_y, epochs=1000)

print("[INFO] evaluating network...")
preds = nn.predict(test_X)
preds = preds.argmax(axis=1)
print(classification_report(test_y.argmax(axis=1), preds))
