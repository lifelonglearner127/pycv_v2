# USAGE
# python augmentation_demo.py --image ~/datasets/cats_01000.jpg --output output

import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory to store augmentation examples")
ap.add_argument("-p", "--prefix", type=str, default="image",
                help="output file prefix")
args = vars(ap.parse_args())

print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2,
                         zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")
total = 0

print("[INFO] generating images...")
image_gen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
                     save_prefix=args["prefix"], save_format="jpg")

for image in image_gen:
    total += 1

    if total == 10:
        break
