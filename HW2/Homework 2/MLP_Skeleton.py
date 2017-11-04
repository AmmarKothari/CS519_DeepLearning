"""
Ammar Kothari
"""


from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np
import pdb
import random
import os, sys
import matplotlib.pyplot as plt

##########################
###### Next step #########
##########################
# check norm of gradient vector and ensure that it is decreasing
# follow trouble shooting steps from his website




# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

	def __init__(self, W, b = 0):
	# DEFINE __init function
	# need size of w - should be tuple?
	# need value of bias input
		self.W = []
		self.b = b
		self.hidden_units = W
		self.velocity_prev = []

	def check_W(self):
		return "W: %s" %self.W

	def forward(self, x):
		if self.W == []:
			print('Initializing W')
			self.W = np.random.randn(np.size(x,0), self.hidden_units)
			# self.W = np.ones((np.size(x,0), self.hidden_units))
			self.velocity_prev = np.zeros(self.W.shape)
			
			if self.W.shape[1] == 1:
				self.W = self.W.flatten()
				self.velocity_prev = self.velocity_prev.flatten()

			self.W = self.W.T
			self.velocity_prev = self.velocity_prev.T

		# DEFINE forward function
		# z is the  dimensions: hidden_units x 1
		# pdb.set_trace()
		a = np.dot(self.W,x)
		return a

	def backprop(self, nodes):
		back = nodes
		return back

	def backward(
		self,
		grad_output, 
		learning_rate=0.0, 
		momentum=0.0, 
		l2_penalty=0.0,
	):
		# pdb.set_trace()
		L2_reg = learning_rate * l2_penalty * self.W
		# pdb.set_trace()
		self.W = self.W - learning_rate*grad_output - momentum*self.velocity_prev - L2_reg
		self.velocity_prev = learning_rate*grad_output
		

		

	# DEFINE backward function
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

	def forward(self, x):
	# DEFINE forward function
		# pdb.set_trace()
		z = np.maximum(0, x)
		return z

	def backprop(self, A):
		back = [1 if x > 0 else 0 for x in A]
		return np.array(back)

	# def backward(
	# 	self, 
	# 	grad_output, 
	# 	learning_rate=0.0, 
	# 	momentum=0.0, 
	# 	l2_penalty=0.0,
	# ):
	# DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):

	def __init__(self):
		self.y = []
		self.z2 = []
		self.eps = 10**-5


	def set_y_data(self, y):
		self.y = y

	def set_z2(self, z2):
		self.z2 = z2

	def forward(self, x):
		# DEFINE forward function
		# sigmoid
		# if x < 10:
		# 	x = 10
		z2 = 1 / (1 + np.exp(-x))
		self.set_z2(z2)
		# pdb.set_trace()
		#Cross Entropy
		# z2 = [o1+self.eps for o1 in z2 if o1 < self.eps]
		# z2 = [o1-self.eps for o1 in z2 if (o1-1), self.eps]
		if z2 < self.eps:
			z2 += self.eps
		elif (1-z2) < self.eps:
			z2 -= self.eps
		if z2 <= 0.5:
			pred = 0
		else:
			pred = 1
		y_star = (self.y + 1)/2
		# if self.y > 0.5:
		# 	L = -np.multiply(y_star, np.log(z2))
		# else:
		# 	L = -np.multiply(1-y_star, np.log(1-z2))
		L = -np.multiply(y_star, np.log(z2)) + -np.multiply(1-y_star, np.log(1-z2))
		# check this out, output from L should be scalar
		# L_total = -sum(L)
		L_total = L
		# pdb.set_trace()
		return L_total, pred

	
	def backprop(self):
		y_star = (self.y + 1)/2
		dL_da2 = -y_star + self.z2
		return dL_da2

	# def backward(
	# 	self, 
	# 	grad_output=0.0, 
	# 	learning_rate=0.0,
	# 	momentum=0.0,
	# 	l2_penalty=0.0
	# ):
		# DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

	def __init__(self, input_dims, hidden_units):
	# INSERT CODE for initializing the network
		self.input_dims = input_dims
		self.output_dims = 1
		self.hidden_units = hidden_units
		self.W1 = LinearTransform(self.hidden_units) #input -> hidden
		self.R1 = ReLU() #hidden layer non-linear transform
		self.W2 = LinearTransform(self.output_dims) #hidden -> output
		self.S1 = SigmoidCrossEntropy()
		self.grad_vect1 = list()
		self.grad_vect2 = list()

	def train(
		self, 
		x_batch, 
		y_batch, 
		learning_rate, 
		momentum,
		l2_penalty,
	):
		mini_batch_size = len(x_batch)
		iteration_count = 0
		max_iter = 10**5
		L_tot = 0.0
		dL_dw2 = 0.0
		dL_dw1 = 0.0
		L_tot = 0.0
		accuracy = 0.0
		anneal = 1
		acc_history = list()
		L_history = list()
		pred_history = list()
		y_history = list()


		for i in range(mini_batch_size):
			list_location = i
			# pdb.set_trace()
			####FORWARD
			self.S1.set_y_data(y_batch[list_location])
			# W1.check_W()
			# pdb.set_trace()
			self.A1 = self.W1.forward(x_batch[list_location])
			# print("shape of A1: %s" %A1.shape)
			self.Z1 = self.R1.forward(self.A1)
			# print("shape of Z1: %s" %Z1.shape)
			self.A2 = self.W2.forward(self.Z1)
			L, pred = self.S1.forward(self.A2)
			pred_history.append(pred)
			y_history.append(y_batch[list_location])
			if pred == y_batch[list_location]:
				accuracy += 1.0
			# pdb.set_trace()
			L_tot += L + (np.linalg.norm(self.W1.W) + np.linalg.norm(self.W2.W))*l2_penalty/2
			if np.isnan(L_tot):
				pdb.set_trace()



			####DERIVATIVES########
			dL_da2  = self.S1.backprop() # delta3*g'
			d3 = dL_da2
			da2_dw2 = self.W2.backprop(self.Z1) #just gives back Z1
			da2_dz1 = self.W2.W
			dz1_da1 = self.R1.backprop(self.A1)
			da1_dw1 = x_batch[list_location]
			# pdb.set_trace()
			dL_dw2 += d3 * da2_dw2.T
			d2 = self.W2.W.T*d3*dz1_da1
			# print("shape of dL_dw2: %s" %dL_dw2.shape)
			part1 = dL_da2 * da2_dz1.T
			part2 = part1 * dz1_da1.T
			# dL_dw1 += np.outer(part2, da1_dw1.T).T
			# pdb.set_trace()
			dL_dw1 += np.outer(d2, x_batch[list_location].T)


		self.grad_vect1.append(np.linalg.norm(dL_dw2))
		self.grad_vect2.append(np.linalg.norm(dL_dw1))
		#######BackProp#########
		dL_dw2 = dL_dw2/mini_batch_size
		dL_dw1 = dL_dw1/mini_batch_size
		# pdb.set_trace()
		self.W2.backward(dL_dw2, learning_rate = learning_rate, momentum = momentum, l2_penalty = l2_penalty)

		self.W1.backward(dL_dw1, learning_rate = learning_rate, momentum = momentum, l2_penalty = l2_penalty)

		learning_rate = max(0.05, learning_rate*anneal)
		L_history.append(L_tot/mini_batch_size)
		acc_history.append(accuracy/mini_batch_size)
		# if abs(accuracy/iter_mini - acc_prev) < 0.01:
		# 	acc_count += 1
		# else:
		# 	acc_count = 0
		# acc_prev = accuracy/iter_mini
		# print('Current Accuracy: %s' %str(accuracy/iter_mini))
		return L_tot




	def evaluate(self, x, y):
		i = 0
	# INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':
	HWset = True
	XORset = False
	if HWset:
		# print 'sys.argv[0] = ', sys.argv[0]
		pathname = os.path.dirname(sys.argv[0])
		# print 'path =', pathname
		fullpath = os.path.abspath(pathname)
		# print 'full path =', fullpath
		cifar_folder = os.path.join(fullpath, 'cifar-2class-py2')
		# print 'cifar folder=', cifar_folder
		cifar_file = os.path.join(cifar_folder, 'cifar_2class_py2.p')
		# print 'cifar file=', cifar_file
		data = cPickle.load(open(cifar_file, 'rb'))

		train_y = data['train_labels']
		test_x = data['test_data']
		test_y = data['test_labels']
		
		train_x_orig = data['train_data']
		mean_x = np.matrix(np.mean(train_x_orig, axis=0))
		std_x  = np.matrix(np.std(train_x_orig, axis=0))
		train_x = np.array((train_x_orig - mean_x)/std_x)

	elif XORset:
		### create xor data ####
		data_set_size = 10000
		x_dims = 2
		train_x = np.random.randint(0,2,size=(data_set_size, x_dims))
		train_y = np.array([0 if x1==x2 else 1 for x1,x2 in train_x])

	num_examples, input_dims = train_x.shape
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	num_epochs = 2 #number of times that all examples are looped through
	num_batches = 1000 #number of gradient descent iterations within epoch
	ind_list = np.arange(num_batches)*(num_examples/num_batches)
	batch_size = num_examples/num_batches
	hidden_units = 1000
	loss_history = list()


	mlp = MLP(input_dims, hidden_units)
	total_loss = 0
	for epoch in xrange(num_epochs):

	# INSERT YOUR CODE FOR EACH EPOCH HERE
	# shuffle indexes
		random.shuffle(ind_list) #shuffles in place
		total_loss = 0
		for b in xrange(num_batches):
			# total_loss = 0.0
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# pdb.set_trace()
			x_batch = train_x[ind_list[b]:ind_list[b]+batch_size]
			y_batch = train_y[ind_list[b]:ind_list[b]+batch_size]
			# pdb.set_trace()
			L_tot = mlp.train(x_batch, y_batch, learning_rate = 0.1, momentum = 0.0, l2_penalty = 0.00)
			loss_history.append(L_tot)
			# pdb.set_trace()
			if HWset:
				total_loss += L_tot[0]
			elif XORset:
				total_loss += L_tot
			# MAKE SURE TO UPDATE total_loss
			print(
				'\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
					epoch + 1,
					b + 1,
					total_loss/b,
				),
				end='',
			)
			sys.stdout.flush()
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
		'''
		print()
		print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
			train_loss,
			100. * train_accuracy,
		))
		print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
			test_loss,
			100. * test_accuracy,
		))
		'''
	plt.plot(loss_history, 'bo') # loss is always positive
	plt.figure()
	plt.subplot(2,1,1)
	plt.plot(mlp.grad_vect1, 'bo')
	plt.subplot(2,1,2)
	plt.plot(mlp.grad_vect2, 'ro')
	plt.show()
	pdb.set_trace()

