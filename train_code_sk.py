
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier

import pickle


def main(epochs=10, save_model=True, seed=1, model_save_path='mymodel.sk'):


	# initializing network
	clf = MLPClassifier(random_state=seed, max_iter=epochs)

	clf = clf.fit(X_train, y_train)
	
	print()

	# save trained model's weights
	if save_model:
		pickle.dump(clf, open(model_save_path, 'wb'))

	return clf


if __name__ == "__main__":

	dataset = 'dataMAMAMIA.csv'
	model_save_path = 'scikit-classifier.pik'

	NUM_CLASSES = 4

	X = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
	y = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

	model = main(epochs=300, seed=1, save_model=True, model_save_path=model_save_path)

	# note for inference:
	# output = np.argmax(clf.predict_proba(X_test), axis=1)