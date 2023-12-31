import numpy as np
from keras.models import load_model
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


def squaredError(y, yhat):
	error = 0
	y = np.resize(y, (3,))
	yhat = np.resize(yhat, (3,))
	for i in range(len(y)):
		error += (yhat[i]-y[i])**2
	return error**(1/2)


def errorCalculator(yhat, img, model):
	saliencyArray = np.zeros((28, 28), dtype=float)
	X = np.zeros((1, 28, 28, 1))
	for i in range(28):
		for j in range(28):
			x = img.copy()
			x[i, j] += 10/255
			x = np.resize(x, (1, 28, 28, 1))
			X = np.append(X, x, axis=0)
	X = X[1:]
	y = model.predict(X)		
	for i in range(28):
		for j in range(28):
			saliencyArray[i, j] = squaredError(y[i*28 + j], yhat)
	return saliencyArray


def saliencyMap(saliencyArray):
	maxim = np.max(saliencyArray)
	saliencyArray = (saliencyArray/maxim)
	return saliencyArray


def main():
	model = load_model("model/DoodleRecognition.h5")

	img = np.zeros((28, 28))
	image = np.resize(img, (1, 28, 28, 1))
	yhat = model.predict(image)

	saliencyArray = errorCalculator(yhat, img, model)
	saliencyArray = saliencyMap(saliencyArray)

	plt.imshow(saliencyArray)
	plt.show()


main()