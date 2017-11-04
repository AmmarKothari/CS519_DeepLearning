"""
INSERT YOUR NAME HERE
"""


from __future__ import division
from __future__ import print_function

import sys

import cPickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
import pickle
import csv
import os


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

	def __init__(self, W, b):
	# DEFINE __init function
		# add one for bias unit
		self.W = W
		self.b = b

	def forward(self, x):
	# DEFINE forward function
		out = np.dot(self.W, x.T)
		# out will be [hidden + 1 x inputs + 1]
		# plus one is from bias units
		return out

	def backward(
		self, 
		grad_output, 
		learning_rate=0.0, 
		momentum=0.0, 
		l2_penalty=0.0,
	):
		self.W += grad_output

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

	def forward(self, x):
	# DEFINE forward function
		out = np.maximum(x, 0)
		return out

	def backward(self, x):
		out = np.maximum(x, 0) # sets all negative to zero
		out = np.minimum(out, 1) # sets all positive to 1
		return out

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
	def __init__(self):
		self.y = 0.0

	def set_y(self, y):
		self.y = y

	def forward(self, x):
		# DEFINE forward function
		#sigmoid

		#dealing with overflow
		if x <= -100:
			x = -100
		nl = 1/(1+np.exp(-x))
		y_star = self.y
		epsilon = 1e-300
		epsilon2 = 0.999999999999999
		if nl <= epsilon:
			nl = epsilon
		if nl >= epsilon2:
			nl = epsilon2

		#L_pre is negative for all values
		L_pre = y_star * np.log(nl) + (1 - y_star) * np.log(1-nl)
		L = -L_pre #makes L positive
		return L, nl

	def backward(self, x):
		# y_star = (1 + self.y)/2
		out = (self.y - x)
		return out

class Sigmoid(object):

	def forward(self, x):
		z = 1/(1+np.exp(-x))
		return z

	def backward(self, x):
		out = np.multiply(x, (1 - x)) #elementwise multiplication
		return out

class LSM(object):
	def __init__(self):
		self.y = 0.0

	def set_y(self, y):
		self.y = y

	def forward(self, x):
		L = 0.5 * (x-self.y)**2
		return L

	def backward(self, x):
		out = abs(x-self.y)
		return out


# This is a class for the Multilayer perceptron
class MLP(object):

	def __init__(self, input_dims, hidden_units):
		# INSERT CODE for initializing the network
		self.input_dims = input_dims
		self.hidden_units = hidden_units
		self.batch_size = 0
		self.epochs = 0
		self.num_batches = 0
		self.bias = 1
		W1_initial = 1 - 2*np.random.rand(self.hidden_units, self.input_dims + 1)
		W2_initial = 1 - 2*np.random.rand(1, self.hidden_units + 1)
		self.W1 = LinearTransform(10*W1_initial, 1)
		self.W2 = LinearTransform(10*W2_initial, 1)
		self.test_accuracy_record = list()
		self.train_accuracy_record = list()
		# self.NL1 = Sigmoid()
		self.NL1 = ReLU()
		self.NL2 = SigmoidCrossEntropy()

		self.accuracy = np.empty((0,2))
		self.acc_p = list()
		self.train_acc_p = list()

		self.W2_vel = 0
		self.W1_vel = 0

		self.file_name = ''

	def new_epoch(self):
		self.W2_vel = 0
		self.W1_vel = 0

	def train(
		self, 
		x_batch, 
		y_batch, 
		learning_rate, 
		momentum,
		l2_penalty):
		# INSERT CODE for training the network

		pred_1 = 0
		pred_0 = 0
		anneal = 0.999
		L_tot = 0
		output_hist = list()
		dL_dW2 = 0.0
		dL_dw1 = 0.0
		for i2 in range(1):
			output_hist_mini = list()
			mini_batch_count = 0
			acc_local = list()
			for i in range(len(y_batch)):
				mini_batch_count += 1
				x_b = np.append(x_batch[i], self.bias)
				L1 = self.W1.forward(x_b)
				L1 = L1.reshape(-1,1)
				nl1 = self.NL1.forward(L1)

				nl1_b = np.append(nl1, self.bias)
				L2 = self.W2.forward(nl1_b)
				self.NL2.set_y(y_batch[i]) #cross entropy
				L, nl2 = self.NL2.forward(L2) #cross entropy
				if nl2 < 0.5:
					pred = 0
				else:
					pred = 1

				acc_local.append((y_batch[i], pred))
				L_tot += L #+ l2_penalty*0.5*(np.linalg.norm(self.W2.W) + np.linalg.norm(self.W1.W))**2

				delta_i = self.NL2.backward(nl2)
				dL_dW2 += delta_i * nl1_b.T

				dL_dw1_c = (self.NL1.backward(nl1).T * (self.W2.W[0][0:-1]*delta_i)).T * x_b
				dL_dw1 +=   dL_dw1_c
				# print('Norm of Gradient for W1: %s' %np.linalg.norm(dL_dw1))


			learning_rate *= anneal

			self.accuracy = np.append(self.accuracy, np.array(acc_local))

			W2_l2_reg = np.linalg.norm(self.W2.W)
			W1_l2_reg = np.linalg.norm(self.W1.W)
			W2_step = learning_rate * dL_dW2/mini_batch_count + learning_rate*momentum*self.W2_vel #+ learning_rate*l2_penalty*W2_l2_reg
			W1_step = learning_rate * dL_dw1/mini_batch_count + learning_rate*momentum*self.W1_vel #+ learning_rate*l2_penalty*W1_l2_reg
			self.W2.backward(W2_step)
			self.W1.backward(W1_step)
			self.W2_vel = W2_step
			self.W1_vel = W1_step

			self.accuracy = self.accuracy.reshape(-1,2)
			p_local = [1 if x1 == x2 else 0 for x1, x2 in self.accuracy]
			
			self.acc_p.append(sum(np.array(p_local))/len(p_local))

			try:
				L_ret = L_tot[0]
			except:
				L_ret = L_tot


			# try:
			# 	idaho = self.acc_p[-1]*100
			# 	boise = L_tot[0]
			# except:
			# 	pdb.set_trace()

		return L_ret, self.acc_p[-1]*100

	def evaluate(self, x, y):
		i = 0
		# INSERT CODE for testing the network
		acc_local = list()
		W1 = self.W1
		W2 = self.W2
		NL1 = self.NL1
		NL2 = self.NL2
		L_tot = 0
		accuracy = np.empty((0,2))
		for i in range(len(y)):

			x_b = np.append(x[i], self.bias)
			L1 = W1.forward(x_b)
			L1 = L1.reshape(-1,1)
			nl1 = NL1.forward(L1)

			nl1_b = np.append(nl1, self.bias)
			L2 = W2.forward(nl1_b)
			NL2.set_y(y[i]) #cross entropy
			L, nl2 = NL2.forward(L2) #cross entropy
			L_tot += L
			if nl2 < 0.5:
				pred = 0
			else:
				pred = 1

			acc_local.append((y[i], pred))

		
		accuracy = np.append(accuracy, np.array(acc_local))
		accuracy = accuracy.reshape(-1,2)
		p_local = [1 if x1 == x2 else 0 for x1, x2 in accuracy]
		acc_perc = sum(np.array(p_local))/len(p_local)
		# print('Accuracy: %s' %acc_perc)
		# acc_perc = 0

		try:
			L_ret = L_tot[0]
		except:
			L_ret = L_tot

		return L_ret, acc_perc

if __name__ == '__main__':
	HWset = True
	XORset = False
	ANDset = False
	ORset = False
	if HWset:
		data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
		
		train_x_orig = data['train_data']
		train_y = data['train_labels']
		test_x = data['test_data']
		test_y = data['test_labels']
		mean_x = np.matrix(np.mean(train_x_orig, axis=0))
		std_x  = np.matrix(np.std(train_x_orig, axis=0))
		train_x = np.array((train_x_orig - mean_x)/std_x)

		# train_x = train_x[0:100]
		# train_y = train_y[0:100]

	if XORset:
		### create xor data ####
		data_set_size = 1000
		x_dims = 2
		train_x = np.random.randint(0,2,size=(data_set_size, x_dims))
		train_y = np.array([0 if x1==x2 else 1 for x1,x2 in train_x])
		# train_y = np.array([0 if x1+x2 < 1 else 1 for x1,x2 in train_x]) ## OR function
		test_x = train_x[0:500]
		test_y = train_y[0:500]

	if ANDset:
		### create and data ####
		data_set_size = 10
		x_dims = 2
		train_x = np.random.randint(0,2,size=(data_set_size, x_dims))
		train_y = np.array([0 if (x1+x2)<2 else 1 for x1,x2 in train_x])
	
	if ORset:
		### create and data ####
		data_set_size = 10
		x_dims = 2
		train_x = np.random.randint(0,2,size=(data_set_size, x_dims))
		train_y = np.array([0 if x1+x2<=1 else 1 for x1,x2 in train_x])


	num_examples, input_dims = train_x.shape
	num_epochs = 10
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	load = False
	H_array = [5, 10, 20, 40, 50, 100]
	lr_array = np.logspace(-5, 2, num=8)
	lr_list = list(['00001', '0001', '001', '01', 'A1', '1', '10', '100'])
	BS_array = [10, 50, 100, 500, 1000, 2000, 5000]
	H_part = np.append(np.arange(0,len(H_array)), 4*np.ones((1,len(lr_array) + len(BS_array))))
	lr_part = np.append(3*np.ones((1,len(H_array))), np.arange(0,len(lr_array)))
	lr_part = np.append(lr_part, 3*np.ones((1,len(BS_array))))
	BS_part = np.append(np.zeros((1,len(H_array) + len(lr_array))), np.arange(0, len(BS_array)))
	H_part = H_part.reshape(-1,1)
	lr_part = lr_part.reshape(-1,1)
	BS_part = BS_part.reshape(-1,1)
	param_ind = np.hstack((H_part, lr_part, BS_part))
	param_ind = param_ind.astype(int)

	for i_H, i_lr, i_BS in param_ind:

		###### Loop Code ######
		# print(i_H, i_lr, i_BS)
		file_name = 'H%s_LR%s_BS%s.txt' %(str(H_array[i_H]), lr_list[i_lr], str(BS_array[i_BS]))
		
		# pdb.set_trace()
		print(file_name)
		# continue
		try:
		    os.remove(file_name)
		except OSError:
		    pass
		
		f = file(file_name, 'wb')
		batch_size = BS_array[i_BS]
		hidden_units = H_array[i_H]
		learning_rate = lr_array[i_lr]
		anneal = 0.9999
		test_accuracy_record = list()
		train_accuracy_record = list()
		num_batches = num_examples/batch_size
		mlp = 0
		mlp = MLP(input_dims, hidden_units)
		mlp.epochs = num_epochs
		mlp.batch_size = batch_size
		mlp.num_batches = num_batches
		# plt.ion()

		ind_list = np.arange(num_batches)*(num_examples/num_batches)
		lr = learning_rate
		for epoch in xrange(num_epochs):

			# INSERT YOUR CODE FOR EACH EPOCH HERE
			random.shuffle(ind_list) #shuffles in place
			total_loss = 0
			mlp.new_epoch()
			for b in xrange(int(num_batches)):
				lr = lr*anneal
				total_loss = 0.0
				# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
				x_batch = train_x[ind_list[b]:ind_list[b]+batch_size]
				y_batch = train_y[ind_list[b]:ind_list[b]+batch_size]
				
				loss, acc = mlp.train(x_batch, y_batch, learning_rate = lr, momentum = 0.8, l2_penalty = 0.000)
				total_loss += loss
				# MAKE SURE TO UPDATE total_loss
				print(
					'\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}, Accuracy = {:.2f}%'.format(
						epoch + 1,
						b + 1,
						total_loss,
						acc
					),
					end='',
				)
				sys.stdout.flush()
			# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
			# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
			train_loss, train_accuracy = mlp.evaluate(train_x ,train_y)
			test_loss, test_accuracy   = mlp.evaluate(test_x, test_y)
			print()
			print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
				train_loss,
				100. * train_accuracy,
			))
			print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
				test_loss,
				100. * test_accuracy,
			))
			mlp.test_accuracy_record.append(test_accuracy)
			mlp.train_accuracy_record.append(train_accuracy)
			# pdb.set_trace()
			with open(file_name, 'a') as csvfile:
				accwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
				accwriter.writerow([train_accuracy, test_accuracy, train_loss, test_loss])



	


	# plt.plot(mlp.acc_p, 'o')
	# plt.plot(mlp.test_accuracy_record, 'o-', label = 'Test Set')
	# plt.plot(mlp.train_accuracy_record, 'o-', label = 'Training Set')
	# plt.legend()
	# plt.ylabel('Accuracy')
	# plt.xlabel('Batch Number')
	# plt.show()
	# pdb.set_trace()
