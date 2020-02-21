# USAGE
# python linear_example.py

import cv2
import numpy as np


labels = ["dog", "cat", "pandas"]
np.random.seed(1)

W = np.random.randn(3, 3072)
b = np.random.randn(3)

orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

scores = W.dot(image) + b

for (label, score) in zip(labels, scores):
    print(f"[INFO] {label}: {score:.2f}")

cv2.putText(orig, f"{labels[np.argmax(scores)]}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey(0)
