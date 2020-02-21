# USAGE
# python annotate.py --input download --annot dataset

import argparse
import os
import cv2
import imutils
from imutils import paths


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
                help="path to output directory of annotations")
args = vars(ap.parse_args())

image_paths = list(paths.list_images(args["input"]))
counts = {}

for (i, image_path) in enumerate(image_paths):
    print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            if key == ord("`"):
                print("[INFO] ignoring character")
                continue

            key = chr(key).upper()
            dir_path = os.path.sep.join([args["annot"], key])

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            count = counts.get(key, 1)
            p = os.path.sep.join([dir_path, f"{str(count).zfill(6)}.png"])
            cv2.imwrite(p, roi)

            counts[key] = count + 1

    except KeyboardInterrupt:
        print("[INFO] manually leaving script")
        break
    except Exception:
        print("[INFO] skipping image...")
