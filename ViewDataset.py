import numpy as np 
import matplotlib.pyplot as plt 

filename = input("Enter the file name without extension: ")
n = int(input("Enter the number of images you want to see: "))

images = np.load('dataset/'+filename + '.npy')

for i in range(n):
	img = np.reshape(images[i], (28, 28))
	plt.imshow(img)
	plt.show()
