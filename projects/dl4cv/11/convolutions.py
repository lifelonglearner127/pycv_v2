# USAGE
# python convolutions.py --image ~/Downloads/1.png

import argparse
import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def convolve(image, K):
    (image_height, image_width) = image.shape[:2]
    (kernel_height, kernel_width) = K.shape[:2]

    pad = (kernel_width - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((image_height, image_width), dtype="float")

    for y in np.arange(pad, image_height + pad):
        for x in np.arange(pad, image_width + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * K).sum()
            output[y - pad, x-pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
args = vars(ap.parse_args())

small_blur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
large_blur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
sharpen = np.array(
    (
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ),
    dtype="int"
)
laplacian = np.array(
    (
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ),
    dtype="int"
)
sobel_X = np.array(
    (
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ),
    dtype="int"
)
sobel_Y = np.array(
    (
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ),
    dtype="int"
)
emboss = np.array(
    (
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ),
    dtype="int"
)

kernel_bank = (
    ("small_blur", small_blur),
    ("large_blur", large_blur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobel_X),
    ("sobel_y", sobel_Y),
    ("emboss", emboss)
)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernel_name, K) in kernel_bank:
    print(f"[INFO] applying {kernel_name} kernel")
    convolve_output = convolve(gray, K)
    opencv_output = cv2.filter2D(gray, -1, K)

    cv2.imshow("Original", gray)
    cv2.imshow(f"{kernel_name} - convolve", convolve_output)
    cv2.imshow(f"{kernel_name} - opencv", opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
