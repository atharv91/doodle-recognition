from keras.preprocessing.image import ImageDataGenerator
from os import listdir
from os.path import isfile, join
import math
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K
import matplotlib.pyplot as plt
import numpy

# Input shape: (28, 28)
# Classes: n
def hand_model(n):
    # num_of_classes = 1
    num_of_classes = n
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))
    # model.add(Dense(num_of_classes, activation='sigmoid'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def saveModel():
    model_json = Model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    Model.save_weights("model.h5")
    print("Saved model to disk")


def takeInput(classes, k):
    TrainX, TrainY = np.zeros((1, 28, 28)), np.zeros((1,), dtype='int8')
    print(classes)
    for i, j in enumerate(classes):
        if j=="":
            continue
        x = np.load('dataset/' + j + '.npy')
        x = x/255
        x = x[120000*(k):120000*(k+1)]
        x = np.reshape(x, (x.shape[0], 28, 28))
        TrainX = np.append(TrainX, x, axis=0)
        y = np.ones((x.shape[0]), dtype='int8') * i
        TrainY = np.append(TrainY, y)
    TrainX, TrainY = TrainX[1:], TrainY[1:]
    return TrainX, TrainY


def augmentData(TrainX, TrainY):
    TrainX = np.append(TrainX, TrainX[:, :, ::-1], axis=0)
    TrainY = np.append(TrainY, -TrainY, axis=0)
    return TrainX, TrainY


def loadClasses(path):
    classes = [f.split(".")[0] for f in listdir(path) if isfile(join("dataset/", f))]
    #print(classes)
    return classes


def main():
    datasetDir = "dataset/"
    classes = loadClasses(datasetDir)
    model = hand_model(len(classes))
    for i in (range(1)):
        TrainX, TrainY = takeInput(classes, i)
        # TrainX, TrainY = augmentData(TrainX, TrainY)
        print(TrainX.shape)
        print(TrainY.shape)

        TrainX, TrainY = shuffle(TrainX, TrainY)


        TrainY = to_categorical(TrainY)
        train_x, test_x, train_y, test_y = train_test_split(TrainX, TrainY, random_state=0, test_size=0.1)
        print(train_y.shape)

        train_x = np.reshape(train_x, (train_x.shape[0], 28, 28, 1))
        test_x = np.reshape(test_x, (test_x.shape[0], 28, 28, 1))

        history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=30, batch_size=32)
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    model.save('model/DoodleRecognition40.h5')

def getScores():
    datasetDir = "dataset/"
    classes = loadClasses(datasetDir)
    TrainX, TrainY = takeInput(classes, 0)
    TrainX, TrainY = shuffle(TrainX, TrainY)
    TrainY = to_categorical(TrainY)
    train_x, test_x, train_y, test_y = train_test_split(TrainX, TrainY, random_state=0, test_size=0.1)
    print(train_y.shape)

    train_x = np.reshape(train_x, (train_x.shape[0], 28, 28, 1))
    test_x = np.reshape(test_x, (test_x.shape[0], 28, 28, 1))
    model = keras.models.load_model('model/DoodleRecognition.h5')
    y_pred = model.predict(test_x, batch_size=64, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(test_y, y_pred_bool))

main()


