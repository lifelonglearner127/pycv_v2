import cv2
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
