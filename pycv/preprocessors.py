import cv2
import imutils
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
