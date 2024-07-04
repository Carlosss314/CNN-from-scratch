import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

#getting and preparing the data
def load_data(train_size, test_size):
    #load train
    train = h5py.File("dataset/train.hdf5", 'r')
    x_train, y_train = train['image'][...], train['label'][...]
    train.close()

    #load test
    test = h5py.File("dataset/test.hdf5", 'r')
    x_test, y_test = test['image'][...], test['label'][...]
    test.close()

    #prendre seulment le nombre d'images voulu
    x_train = x_train[:train_size, :, :] / 255
    y_train = y_train[:train_size]
    x_test = x_test[:test_size, :, :] / 255
    y_test = y_test[:test_size]

    return x_train, y_train, x_test, y_test

def model_parameters(fichier):
    w = np.load(f"{fichier}/w.npy")
    b = np.load(f"{fichier}/b.npy")
    filter = np.load(f"{fichier}/filter.npy")
    return w, b, filter


#functions needed to make predictions on the test set
def conv(input_img, filter):
    feature_maps = np.zeros((input_img.shape[0]-filter.shape[0]+1, input_img.shape[0]-filter.shape[0]+1, filter.shape[2]))

    for p in range(filter.shape[2]):
        for i in range(feature_maps.shape[0]):
            for j in range(feature_maps.shape[1]):
                    if (i+3<feature_maps.shape[0] and j+3<feature_maps.shape[1]):
                        patch_of_img = input_img[i:i+3,j:j+3]
                        patch=np.multiply(patch_of_img, filter[:,:,p])
                        feature_maps[i,j,p]=np.sum(patch)

    return feature_maps

def ReLU(feature_maps):
    return np.maximum(feature_maps, 0)

def softmax(z):
    return np.exp(z - max(z))/np.sum(np.exp(z - max(z)))

def forward_prop(img, filter, w, b):
    feature_maps = conv(img, filter)
    feature_maps_relu = ReLU(feature_maps)

    z = np.zeros((w.shape[0], 1))
    for i in range(w.shape[0]):
        z[i] = np.sum(np.multiply(w[i,:,:,:], feature_maps_relu)) + b[i]

    a = softmax(z)
    prediction = np.argmax(a)

    return prediction




x_train, y_train, x_test, y_test = load_data(0, 100)
w, b, filter = model_parameters("weights_and_bias2")


predictions = []
for i in tqdm(range(len(x_test))):
    predicted_value = forward_prop(x_test[i], filter, w, b)
    predictions.append(predicted_value)




#display some values
for i in range(0, 30):
    plt.imshow(x_test[i].reshape(28, 28), cmap="Greys")
    plt.title(f"prediction: {predictions[i]}    actual number: {y_test[i]}")
    plt.show()