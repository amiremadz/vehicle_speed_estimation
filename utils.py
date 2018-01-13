import numpy as np
from keras.applications import vgg16
from PIL import Image
import h5py
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model


class MyImage(object):
    def __init__(self, img_path, dims=None):
        self.img_path = img_path
        self.dims = dims

    def __img2array(self):
        """
        Util function for loading RGB image into 3D numpy array.
        Returns array of shape (H, W, C)

        References
        ----------
        - adapted from keras preprocessing/image.py
        """
        img = Image.open(self.img_path)
        img = img.convert('RGB')
        if self.dims:
            width  = self.dims[1]
            height = self.dims[0]
            img = img.resize((width, height))
            x = np.asarray(img, dtype='float32')
        return x

    def conv_jpg2array(self):
        """
        Loads image using img_to_array, expands it to 4D tensor
        of shape (1, H, W, C), preprocesses it for use in the
        VGG16 network and resequeezes it to a 3D tensor.
    
        References
        ----------
        - adapted from keras preprocessing/image.py
        """
        img = self.__img2array()
        img = np.expand_dims(img, axis=0)
        img = vgg16.preprocess_input(img)
        img = np.squeeze(img)
        return img


class MyHDF5(object):
    def __init__(self, infile=None, outfile=None):
        self.infile = infile
        self.outfile = outfile

    def write(self, arr):
        """
        Write a numpy array to a HDF5 file
        """
        with h5py.File(self.outfile, "w", libver="latest") as fh:
            fh.create_dataset("image", data=arr, dtype=arr.dtype)

    def load(self):
        """
        Load a numpy array stored in HDF5 format into a numpy array
        """
        with h5py.File(self.infile, "r", libver="latest") as fh:
            return fh["image"][:]

def load_data(hdf5_path):
    hdf5_manip = MyHDF5()
    
    hdf5_manip.infile = os.path.join(hdf5_path, "X_train.hdf5")
    X_train = hdf5_manip.load()

    hdf5_manip.infile = os.path.join(hdf5_path, "y_train.hdf5")
    y_train = hdf5_manip.load()

    hdf5_manip.infile = os.path.join(hdf5_path, "X_test.hdf5")
    X_test = hdf5_manip.load()

    hdf5_manip.infile = os.path.join(hdf5_path, "y_test.hdf5")
    y_test = hdf5_manip.load()

    return X_train, y_train, X_test, y_test 


def get_features(hdf5_path, dnn=None, layer=None):
    """
    Computes features of frames from given layer of 
    network
    
    Wrties them to HDF5 files when done.
    """

    # laod the data
    X_train, y_train, X_test, y_test = load_data(hdf5_path)

    print("X_train: {0}".format(X_train.shape))
    print("X_test: {0}".format(X_test.shape))

    base_model = VGG16(weights='imagenet', include_top=False)

    # print layer names
    #for i, layer in enumerate(model.layers):
    #    print(i, layer.name)

    model = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)

    print("Computing train features...")
    train_features = model.predict(X_train)
    print("Train features shape: {0}".format(train_features.shape))
    train_features = np.reshape(train_features, (train_features.shape[0], -1))
    print("Train features shape: {0}".format(train_features.shape))
