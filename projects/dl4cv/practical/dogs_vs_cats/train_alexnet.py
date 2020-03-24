import datetime
import json
import os
import matplotlib
matplotlib.use("Agg")

import config

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from pycv.callbacks import TrainingMonitor
from pycv.datasets import HDF5DatasetGenerator
from pycv.networks import AlexNet
from pycv.preprocessors import ImageToArrayPreProcessor
from pycv.preprocessors import MeanPreProcessor
from pycv.preprocessors import PatchPreProcessor
from pycv.preprocessors import ResizePreProcessor


aug = ImageDataGenerator(
    rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.15, horizontal_flip=True,
    fill_mode="nearest"
)

means = json.loads(open(config.DATASET_MEAN).read())

rp = ResizePreProcessor(227, 227)
pp = PatchPreProcessor(227, 227)
mp = MeanPreProcessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreProcessor()

train_generator = HDF5DatasetGenerator(
    config.TRAIN_HDF5, 128, aug=aug, preprocessors=[pp, mp, iap], classes=2
)
val_generator = HDF5DatasetGenerator(
    config.VAL_HDF5, 128, preprocessors=[rp, mp, iap], classes=2
)

print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

before = datetime.datetime.now()
model.fit(
    train_generator.generator(),
    steps_per_epoch=train_generator.num_images // 128,
    validation_data=val_generator.generator(),
    validation_steps=val_generator.num_images // 128,
    epochs=75, max_queue_size=0, callbacks=callbacks, verbose=1
)
after = datetime.datetime.now()
execution_time = after - before
print(f"[INFO] It took {execution_time.total_seconds()} to train the network")

print("[INFO] serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

train_generator.close()
val_generator.close()
