import cv2
import imutils


def preprocess(image, width, height):
    """Resize and pad image while preserving the aspect ratio

    Key Arguments:
    image: The input image that we are going to pad and resize.
    width: The target output width of the image.
    height: The target output height of the image.
    """
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)
    else:
        image = imutils.resize(image, height=height)

    pad_width = (width - image.shape[1]) // 2
    pad_height = (height - image.shape[0]) // 2

    image = cv2.copyMakeBorder(image, pad_height, pad_width,
                               pad_height, pad_width, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image
