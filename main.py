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

test_img_path = "/home/aemadzadeh/Documents/comma.ai/data/test_results/frames"
test_out_path = "/home/aemadzadeh/Documents/comma.ai/data/test_results/outputs"

def main():
    #load data
    dnn = VGG16
    #dnn = ResNet50
    #dnn = VGG19
    
    X_train, y_train, X_test, y_test = load_data(hdf5_path)
   
    # if features are not obtained yet
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

    train_model = False
    alpha = 15.0

    if train_model:
        print("Training linear regression model...")
        clf = Ridge(alpha=alpha)
        tic = time.time()
        clf.fit(train_features, y_train)
        toc = time.time()
        print("Time elapsed: {0} seconds".format(toc - tic))

        print("Predicting...")
        predictions = clf.predict(test_features)

        mse = metrics.mean_squared_error(y_test, predictions)
        print("MSE: {0}".format(mse))

    pickle_manip = MyPickle()
    pickle_manip.pfile = os.path.join(pickle_path, dnn.__name__, "clf_all_" + str(alpha)  + ".sav")

    if train_model:
        # Store the model in a pickle file
        pickle_manip.dump(clf)

    clf = pickle_manip.load()
    print("\nRidge model: ")
    print(clf.get_params())
    print(clf.score(train_features, y_train))

    print("\nModel on Train data:")
    pred = clf.predict(train_features)
    print(pred)
    print(metrics.mean_squared_error(y_train, pred))

    print("\nModel on Test data:")
    predictions = clf.predict(test_features)
    print(predictions)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("\nMSE: {0}".format(mse))

    #### Test results
    
    extract_features = False

    if extract_features:
        # image parameters
        width  = 50
        height = 50
        channels = 3

        # grab all filenames
        extensions = [".jpg"]
        file_names = [fn for fn in os.listdir(test_img_path) if any(fn.endswith(ext) for ext in extensions)]

        # initialize image array which holds frames
        num_imgs = len(file_names)
        X = np.empty((num_imgs, height, width, channels), dtype='float32')

        print("\nConverting jpegs to numpy arrays...\n")

        img_size = (height, width)

        for idx in range(num_imgs):
            if idx % 1000 == 0:
                print("Converting image {0}".format(file_names[idx]))
            file_path = os.path.join(test_img_path, file_names[idx])
            img_manip = MyImage(file_path, img_size)
            img = img_manip.conv_jpg2array()
            X[idx] = img

        print("\nWriting X to HDF5...")
        hdf5_manip = MyHDF5()

        hdf5_manip.hfile = os.path.join(test_out_path, "X.hdf5")
        hdf5_manip.write(X)

        base_model = dnn(weights='imagenet', include_top=False)

        model = Model(input=base_model.input, output=base_model.get_layer('block2_pool').output)

        print("Computing X features...")
        X_features = model.predict(X)
        print("X features shape, before reshape: {0}".format(X_features.shape))
        X_features = np.reshape(X_features, (X_features.shape[0], -1))
        print("X features shape, after reshape: {0}".format(X_features.shape))
 
        print("Writing X feature to HDF5 files...")
        hdf5_manip = MyHDF5()

        hdf5_manip.hfile = os.path.join(test_out_path, dnn.__name__, "X_features.hdf5")
        hdf5_manip.write(X_features)
    else:
        print("Loading X feature from HDF5 files...")
        hdf5_manip = MyHDF5()
    
        hdf5_manip.hfile = os.path.join(test_out_path, dnn.__name__, "X_features.hdf5")
        X_features = hdf5_manip.load()

    print(X_features.shape)
    predictions = clf.predict(X_features)
    print(predictions) 


if __name__ == "__main__":
    main()

  
