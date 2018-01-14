#!/usr/bin/python
import os
import numpy as np

from sklearn.utils import shuffle
from utils import *
 

gt_path  = "/home/aemadzadeh/Documents/comma.ai/data/"
out_path = "/home/aemadzadeh/Documents/comma.ai/data/outputs"
img_path = "/home/aemadzadeh/Documents/comma.ai/data/frames"

# image parameters
width  = 50
height = 50
channels = 3

# grab all filenames
extensions = [".jpg"]
file_names = [fn for fn in os.listdir(img_path) if any(fn.endswith(ext) for ext in extensions)]

# initialize image array which holds frames
num_imgs = len(file_names)
X = np.empty((num_imgs, height, width, channels), dtype='float32')

print("\nConverting jpegs to numpy arrays...\n")

img_size = (height, width)

for idx in range(num_imgs):
    if idx % 1000 == 0:
        print("Converting image {0}".format(file_names[idx]))
    file_path = os.path.join(img_path, file_names[idx])
    img_manip = MyImage(file_path, img_size)
    img = img_manip.conv_jpg2array()
    X[idx] = img

# Read velocity ground truth
with open(os.path.join(gt_path, "train.txt")) as fh:
    y = fh.readlines()

y = [float(val.strip()) for val in y]
y = np.expand_dims(y, axis=1)

###########
num_imgs = len(y)
X = X[:num_imgs]
###########

# create mask for splitting data
num_train = int(0.9 * num_imgs)
num_test  = num_imgs - num_train

train_mask = np.arange(num_train)
test_mask  = np.arange(num_train, num_imgs)


# Shuffle
print("\nShuffling the data...")
X, y = shuffle(X, y, random_state=16)

# Split data into train and test
print("\nSplitting X into train and test...")
X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]


print("\nWriting X-train to HDF5...")
hdf5_manip = MyHDF5()

hdf5_manip.hfile = os.path.join(out_path, "X_train.hdf5")
hdf5_manip.write(X_train)

hdf5_manip.hfile = os.path.join(out_path, "X_test.hdf5")
hdf5_manip.write(X_test)

hdf5_manip.hfile = os.path.join(out_path, "y_train.hdf5")
hdf5_manip.write(y_train)

hdf5_manip.hfile = os.path.join(out_path, "y_test.hdf5")
hdf5_manip.write(y_test)

print("\n Done!")
