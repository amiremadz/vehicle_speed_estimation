import numpy as np
from keras.applications import vgg16
from PIL import Image


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


