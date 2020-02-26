import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pycv.networks import MiniVGGNet


matplotlib.use("Agg")


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory")
ap.add_argument("-m", "--models", required=True,
                help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
                help="# of models to train")
args = vars(ap.parse_args())

((train_X, train_y), (test_X, test_y)) = cifar10.load_data()
train_X = train_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

lb = LabelBinarizer()
train_y = lb.fit_transform(train_y)
test_y = lb.transform(test_y)

label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

aug = ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, fill_mode="nearest"
)

for i in np.arange(0, args["num_models"]):
    print("[INFO] training model {}/{}".format(i + 1, args["num_models"]))
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(
        loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )

    H = model.fit(
        aug.flow(train_X, train_y, batch_size=64),
        validation_data=(test_X, test_y), epochs=40,
        steps_per_epoch=len(train_X) // 64, verbose=1
    )

    p = [args["models"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    predictions = model.predict(test_X, batch_size=64)
    report = classification_report(
        test_y.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=label_names
    )

    p = [args["output"], "model_{}.txt".format(i)]
    f = open(os.path.sep.join(p), "w")
    f.write(report)
    f.close()

    p = [args["output"], "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy for model {}".format(i))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()
