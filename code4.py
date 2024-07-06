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



#deep learning
def initialisation():
    w = np.random.randn(10, 26, 26, 5) / np.sqrt(26)
    b = np.zeros((10, 1)) / np.sqrt(10)
    filter = np.random.randn(3, 3, 5) / np.sqrt(3)
    return w, b, filter

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

    return prediction, a, feature_maps_relu, feature_maps

def convert_y(y):
    arr = np.zeros((10, 1))
    arr[y] = 1
    return arr

def deriv_ReLU(z):
    return np.where(z>0, 1, 0)

def backward_prop(a, y, w, feature_maps_relu, feature_maps, x):
        dz = a - convert_y(y)
        db = dz

        dw = np.zeros(w.shape)
        for i in range(dw.shape[0]):
            dw[i,:,:,:] = dz[i] * feature_maps_relu

        delta = np.zeros(feature_maps_relu.shape)
        for i in range(feature_maps_relu.shape[0]):
            for j in range(feature_maps_relu.shape[1]):
                for p in range(feature_maps_relu.shape[2]):
                    delta[i,j,p]=np.sum(np.multiply(dz, w[:,i,j,p]))

        grad_Zdel = np.multiply(deriv_ReLU(feature_maps), delta)
        dfilter = conv(x, grad_Zdel)

        return dw, db, dfilter

def update_params(w, b, filter, dw, db, dfilter, lr):
    w = w - lr*dw
    b = b - lr*db
    filter = filter - lr*dfilter
    return w, b, filter


def test_accuracy(x_test, y_test, filter, w, b):
    total_correct = 0

    for n in range(len(x_test)):
        x = x_test[n][:]
        y = y_test[n]

        predicted_value, a, feature_maps_relu, feature_maps = forward_prop(x, filter, w, b)

        if (predicted_value == y): total_correct += 1

    return total_correct / len(x_test)


def neural_network(x_train, y_train, x_test, y_test, num_epochs, lr):
    w, b, filter = initialisation()

    train_accuracy_list = []
    test_accuracy_list = []

    for epochs in tqdm(range(num_epochs)):
        total_correct = 0

        for n in tqdm(range(len(x_train))):
            x = x_train[n][:]
            y = y_train[n]

            predicted_value, a, feature_maps_relu, feature_maps = forward_prop(x, filter, w, b)
            dw, db, dfilter = backward_prop(a, y, w, feature_maps_relu, feature_maps, x)
            w, b, filter = update_params(w, b, filter, dw, db, dfilter, lr)

            if (predicted_value == y): total_correct += 1

        #compute accuracy
        accuracy_on_train_set = total_correct / len(x_train)
        print(f"Train accuracy for epoch {epochs+1} : {accuracy_on_train_set}")
        train_accuracy_list.append(accuracy_on_train_set)

        accuracy_on_test_set = test_accuracy(x_test, y_test, filter, w, b)
        print(f"Test accuracy for epoch {epochs+1} : {accuracy_on_test_set}, \n")
        test_accuracy_list.append(accuracy_on_test_set)


    print("final train accuracy: ", train_accuracy_list[-1])
    print("final test accuracy: ",  test_accuracy_list[-1])

    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy_list, label="train cost")
    plt.legend()
    plt.ylim(ymin=0, ymax=1)
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracy_list, label="test cost", color="orange")
    plt.legend()
    plt.ylim(ymin=0, ymax=1)
    plt.show()


    return w, b, filter




x_train, y_train, x_test, y_test = load_data(60000, 1000)
w, b, filter = neural_network(x_train, y_train, x_test, y_test, 3, 0.01)




# #save weights and bias in files
# np.save("weights_and_bias2/w", w)
# np.save("weights_and_bias2/b", b)
# np.save("weights_and_bias2/filter", filter)
