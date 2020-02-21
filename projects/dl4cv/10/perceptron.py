# USAGE
# python perceptron.py

import argparse
import numpy as np
from pycv.networks import Perceptron


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", default="or", help="Dataset type")
args = vars(ap.parse_args())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

if args["mode"] == "or":
    y = np.array([[0], [1], [1], [1]])
elif args["mode"] == "and":
    y = np.array([[0], [0], [0], [1]])
elif args["mode"] == "xor":
    y = np.array([[0], [1], [1], [0]])

print("[INFO] training perceptron...")
p = Perceptron(X.shape[1])
p.fit(X, y, epochs=20)

print("[INFO] testing perceptron...")
for (x, target) in zip(X, y):
    pred = p.predict(x)
    print(f"[INFO] data={x}, ground-truth={target[0]}, pred={pred}")
