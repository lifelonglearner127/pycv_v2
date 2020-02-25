import argparse
import os
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from pycv.datasets import DatasetLoader
from pycv.networks import FCHeadNet
from pycv.preprocessors import AspectAwarePreProcessor
from pycv.preprocessors import ImageToArrayPreProcessor


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
                help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest"
)

print("[INFO] loading images...")
image_paths = list(paths.list_images(args["dataset"]))
class_names = [pt.split(os.path.sep)[-2] for pt in image_paths]
class_names = [str(x) for x in np.unique(class_names)]

aap = AspectAwarePreProcessor(224, 224)
iap = ImageToArrayPreProcessor()

dataset_loader = DatasetLoader(preprocessors=[aap, iap])
(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(train_X, test_X, train_y, test_y) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

train_y = LabelBinarizer().fit_transform(train_y)
test_y = LabelBinarizer().fit_transform(test_y)

base_model = VGG16(weights="imagenet", include_top=False,
                   input_tensor=Input(shape=(224, 224, 3)))
head_model = FCHeadNet.build(base_model, len(class_names), 256)
model = Model(inputs=base_model.input, outputs=head_model)

for layer in base_model.layers:
    layer.trainable = False

print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

print("[INFO] training head...")
model.fit(
    aug.flow(train_X, train_y, batch_size=32),
    validation_data=(test_X, test_y), epochs=25,
    steps_per_epoch=len(train_X) // 32, verbose=1
)

print("[INFO] evaluating after initialization...")
predictions = model.predict(test_X, batch_size=32)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=class_names
    )
)

for layer in base_model.layers[15:]:
    layer.trainable = True

print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

print("[INFO] training head...")
model.fit(
    aug.flow(train_X, train_y, batch_size=32),
    validation_data=(test_X, test_y), epochs=100,
    steps_per_epoch=len(train_X) // 32, verbose=1
)

print("[INFO] evaluating after initialization...")
predictions = model.predict(test_X, batch_size=32)
print(
    classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=class_names
    )
)

print("[INFO] serializing model...")
model.save(args["model"])
