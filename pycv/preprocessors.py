import cv2
import imutils
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from tensorflow.keras.preprocessing.image import img_to_array


class ResizePreProcessor:

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interpolation)


class ImageToArrayPreProcessor:

    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


class AspectAwarePreProcessor:

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        d_w = 0
        d_h = 0

        if w < h:
            image = imutils.resize(image, width=self.width,
                                   inter=self.interpolation)
            d_h = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height,
                                   inter=self.interpolation)
            d_w = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
        image = image[d_h:h - d_h, d_w:w - d_w]

        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interpolation)


class MeanPreProcessor:

    def __init__(self, r_mean, g_mean, b_mean):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype("float32"))

        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        return cv2.merge([B, G, R])


class PatchPreProcessor:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width),
                                  max_patches=1)[0]


class CropPreProcessor:

    def __init__(self, width, height, horiz=True,
                 interpolation=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.interpolation = interpolation

    def preprocess(self, image):
        crops = []

        (h, w) = image.shape[:2]
        coords = [
            [0, 0, self.width, self.height],
            [w - self.width, 0, w, self.height],
            [w - self.width, h - self.height, w, h],
            [0, h - self.height, self.width, h]
        ]

        d_w = int(0.5 * (w - self.width))
        d_h = int(0.5 * (h - self.height))
        coords.append([d_w, d_h, w - d_w, h - d_h])

        for (start_x, start_y, end_x, end_y) in coords:
            crop = image[start_y:end_y, start_x:end_x]
            crop = cv2.resize(crop, (self.width, self.width),
                              interpolation=self.interpolation)
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)
