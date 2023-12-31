import numpy as np
from keras.models import load_model
from os import listdir
from os.path import isfile, join
import pickle

def loadClasses(path):
    classes = [f.split(".")[0] for f in listdir(path) if isfile(join("dataset/", f))]
    return classes


def main():
    classDir = 'dataset/'
    classes = loadClasses(classDir)
    p = np.load('intermediate/Image.npy')
    p = p / 255
    '''
    filename = 'model/SVMmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    x = np.resize(p, (1, 784))
    svmPredict = loaded_model.predict(x)

    filename = 'model/SVMmodel+.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    svm2Predict = loaded_model.predict(x)
    for i in range(svm2Predict.shape[0]):
    	svm2Predict[i] = int(svm2Predict[i]/3)

    filename = 'model/KNNModel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    knnPredict = loaded_model.predict(x)
    '''
    model = load_model("model/DoodleRecognition.h5")
    y = np.resize(p, (1, 28, 28, 1))

    cnnPredict = model.predict(y)

    '''
    print('\n\n\n')
    print(classes)
    print('SVM prediction: ' + classes[svmPredict[0]])
    print('SVM+ prediction: ' + classes[svm2Predict[0]])
    print('KNN prediction: ' + classes[knnPredict[0]])
    '''
    print(cnnPredict[0])
    print('CNN prediction: ' + classes[int(np.argmax(cnnPredict[0]))] + " with " + str(round(np.max(cnnPredict[0]) * 100, 2)) + ' percent confidence')


main()