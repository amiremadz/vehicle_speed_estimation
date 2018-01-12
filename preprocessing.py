#!/usr/bin/python
import os
import numpy as np

from sklearn.utils import shuffle
from utils import MyImage


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


# create mask for splitting data
num_imgs = len(file_names)

num_train = int(0.9 * num_imgs)
num_test  = num_imgs - num_train

mask_train = np.arange(num_train)
mask_test  = np.arange(num_train, num_imgs)

# initialize image array which holds frames
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



