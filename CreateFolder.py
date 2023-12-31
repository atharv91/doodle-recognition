import os
direc = ["KMeanY", "dataset", "model", "intermediate"]
for d in direc:
	if not os.path.exists(d):
		os.makedirs(d)