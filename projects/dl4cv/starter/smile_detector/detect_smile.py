# USAGE
# python detect_smile.py --cascade haarcascade.xml --model output/lenet.hdf5

import argparse
import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
                help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (f_x, f_y, f_w, f_h) in rects:
        roi = gray[f_y:f_y + f_h, f_x:f_x + f_w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (not_smiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > not_smiling else "Not Smiling"

        cv2.putText(frame_clone, label, (f_x, f_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (f_x, f_y), (f_x + f_w, f_y + f_h),
                      (0, 0, 255), 2)

    cv2.imshow("Face", frame_clone)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
