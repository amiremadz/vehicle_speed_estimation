from utils import * 

hdf5_path = "/home/aemadzadeh/Documents/comma.ai/data/outputs"




def main():
    #load data
    X_train, y_train, X_test, y_test = load_data(hdf5_path)
    
    get_features(hdf5_path);
        
        



if __name__ == "__main__":
    main()
