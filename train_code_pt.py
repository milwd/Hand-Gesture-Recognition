
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.d1 = nn.Linear(21*2, 60)
		self.d2 = nn.Linear(60, 40)
		self.d3 = nn.Linear(40, NUM_CLASSES)

	def forward(self, x):
		x = self.d1(x)
		x = F.relu(x)
		x = self.d2(x)
		x = F.relu(x)
		x = self.d3(x)
		return x

def main(batch_size=128, test_batch_size=1000, epochs=10, lr=0.001, no_cuda=True, seed=None, save_model=True, model_save_path='mymodel.pt'):

	test_losses = []

	# random seed (if needed)
	if seed:
		torch.manual_seed(seed)

	# cuda device initialization
	use_cuda = not no_cuda and torch.cuda.is_available()
	if use_cuda:
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	# setup PyTorch's data loader for ease 
	train_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.DoubleTensor(X_train),torch.DoubleTensor(y_train)),batch_size=batch_size, shuffle=True)
	test_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.DoubleTensor(X_test), torch.DoubleTensor(y_test)),shuffle=True)

	# initializing network
	model = Net().to(device)

	# setup network managers : optimizer and loss function
	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_function = nn.CrossEntropyLoss()

	# training loop
	for epoch in range(1, epochs + 1):
		for batch_idx, (data, target) in enumerate(train_loader):  # split data into batches with data loaders
			data, target = data.to(device), target.to(device)
			# make optimizer ready for a data input and backprop
			optimizer.zero_grad()
			# input the data into the network and get the result
			output = model(data.float())
			# calculate loss for the input data and target(y)
			loss = loss_function(output, target.long())
			# perform backpropagation
			loss.backward()
			optimizer.step()

		# get loss/accuracy for each epoch
		model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data.float())
				test_loss += loss_function(output, target.long()).item()
				pred = output.argmax(dim=1, keepdim=True) 
				correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(test_loader.dataset)
		test_losses.append(test_loss)

		print('\nepoch: {}\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch, test_loss, correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))

	# save trained model's weights
	if save_model:
		torch.save(model.state_dict(), model_save_path)

	plt.plot(test_losses)
	plt.show()

	return model


if __name__ == "__main__":

	dataset = 'dataMAMAMIA.csv' #'model/keypoint_classifier/keypoint.csv'
	model_save_path = 'pytorch-classifier.pt'

	NUM_CLASSES = 5

	X = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
	y = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

	model = main(batch_size=64, test_batch_size=1000, epochs=500, lr=0.001, no_cuda=True, seed=1, save_model=True, model_save_path=model_save_path)

	# note for inference:
	# output = np.argmax(model(torch.Tensor[-]).detach().numpy())