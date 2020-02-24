import argparse
import pickle
import h5py
from pycv.helpers import rank5_accuracy


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True,
                help="path HDF5 database")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

print("[INFO] loading pre-trained model...")
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

print("[INFO] predicting...")
preds = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(preds, db["labels"][i:])

print(f"[INFO] rank-1: {rank1 * 100:.2f}%")
print(f"[INFO] rank-5: {rank5 * 100:.2f}%")
db.close()
