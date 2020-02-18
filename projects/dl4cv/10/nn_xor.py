# USAGE
# python nn_xor.py

import numpy as np
from pycv.networks import NeuralNetwork


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, epochs=20000)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"[INFO] data={x}, ground-truth={target[0]},"
          f" pred={pred:.4f}, step={step}")
