from utils import * 
import os
import time
from sklearn.linear_model import Ridge
from sklearn import metrics
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50

hdf5_path = "/home/aemadzadeh/Documents/comma.ai/data/outputs"
pickle_path = hdf5_path

def main():
    #load data
    #dnn = VGG16
    dnn = ResNet50
    #dnn = VGG19
    
    
    X_train, y_train, X_test, y_test = load_data(hdf5_path)
    
    #get_features(hdf5_path, dnn, 'block2_pool');

    print("Loading feature from HDF5 files...")
    hdf5_manip = MyHDF5()
    
    hdf5_manip.hfile = os.path.join(hdf5_path, dnn.__name__, "train_features.hdf5")
    train_features = hdf5_manip.load()

    hdf5_manip.hfile = os.path.join(hdf5_path, dnn.__name__, "test_features.hdf5")
    test_features = hdf5_manip.load()

    hdf5_manip.hfile = os.path.join(hdf5_path, "y_train.hdf5")
    y_train = hdf5_manip.load()

    hdf5_manip.hfile = os.path.join(hdf5_path, "y_test.hdf5")
    y_test = hdf5_manip.load()

    print("Training linear regression model...")
    clf = Ridge(alpha=5.0)
    tic = time.time()
    clf.fit(train_features, y_train)
    toc = time.time()
    print("Time elapsed: {0} seconds".format(toc - tic))

    #print("Predicting...")
    #predictions = clf.predict(test_features)

    #mse = metrics.mean_squared_error(y_test, predictions)
    #print("MSE: {0}".format(mse))

    # Store the model in a pickle file
    pickle_manip = MyPickle()
    pickle_manip.pfile = os.path.join(pickle_path, dnn.__name__, "clf_all.sav")
    
    #pickle_manip.dump(clf)

    clf = pickle_manip.load()
    predictions = clf.predict(test_features)

    mse = metrics.mean_squared_error(y_test, predictions)
    print("MSE: {0}".format(mse))

if __name__ == "__main__":
    main()

  
