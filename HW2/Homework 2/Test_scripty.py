import cPickle
import numpy as np
import os, sys
import pdb
import matplotlib.pyplot as plt
import time

import MLP_Skeleton as MLP

start_time = time.time()
PlotOn = True
# print 'sys.argv[0] =', sys.argv[0]
# pathname = os.path.dirname(sys.argv[0])
# print 'path =', pathname
# fullpath = os.path.abspath(pathname)
# print 'full path =', fullpath
# cifar_folder = os.path.join(fullpath, 'cifar-2class-py2')
# print 'cifar folder=', cifar_folder
# cifar_file = os.path.join(cifar_folder, 'cifar_2class_py2.p')
# print 'cifar file=', cifar_file


# data = cPickle.load(open(cifar_file, 'rb'))
# train_x_orig = data['train_data']
# mean_x = np.matrix(np.mean(train_x_orig, axis=0))
# std_x  = np.matrix(np.std(train_x_orig, axis=0))
# train_x = np.array((train_x_orig - mean_x)/std_x)	
# train_y = data['train_labels']
# test_x = data['test_data']
# test_y = data['test_labels']

### create xor data ####
data_set_size = 10000
x_dims = 2
train_x = np.random.randint(0,2,size=(data_set_size, x_dims))
train_y = [0 if x1==x2 else 1 for x1,x2 in train_x]
# train_y = [0 if sum(train_x[i])<=x_dims/2 else 1 for i,x in enumerate(train_x)]

test_set_size = 100
test_x = np.random.randint(0,2,size=(test_set_size, x_dims))
test_y = [0 if x1==x2 else 1 for x1,x2 in train_x]
# test_y = [0 if sum(test_x[i])<=x_dims/2 else 1 for i,x in enumerate(test_x)]


n_train_ex = len(train_x)
hidden_units = 4
L_tot = 10**5
iteration_count = 0
mini_batch = 100
max_iter = 10**5
accuracy = 0.0
acc_count = 0.0
acc_prev = 0.0
alpha = 0.1
# anneal = 0.99
anneal = 1
l2_penalty = 0.01


acc_history = list()
L_history = list()
pred_history = list()
y_history = list()

#set network params
W1 = MLP.LinearTransform(hidden_units)
W2 = MLP.LinearTransform(1)
R1 = MLP.ReLU()
S1 = MLP.SigmoidCrossEntropy()
plt.ion()
# pdb.set_trace()
while acc_count < 10 and iteration_count<max_iter:
	L_tot = 0.0
	dL_dw2 = 0.0
	dL_dw1 = 0.0
	iter_mini = 0.0
	accuracy = 0.0
	for i in range(mini_batch):
	# for i in range(n_train_ex):
		iteration_count += 1
		iter_mini += 1
		# list_location = iteration_count % n_train_ex
		list_location = np.random.randint(0, n_train_ex)
		#forward
		S1.set_y_data(train_y[list_location])
		# W1.check_W()
		A1 = W1.forward(train_x[list_location])
		# print("shape of A1: %s" %A1.shape)
		Z1 = R1.forward(A1)
		# print("shape of Z1: %s" %Z1.shape)
		A2 = W2.forward(Z1)
		L, pred = S1.forward(A2)
		# pdb.set_trace()
		# print('Predicted Value: %s' %S1.z2)
		pred_history.append(pred)
		y_history.append(train_y[list_location])
		# pdb.set_trace()
		if pred == train_y[list_location]:
			accuracy += 1.0
		# pdb.set_trace()
		L_tot += L + (np.linalg.norm(W1.W) + np.linalg.norm(W2.W))*l2_penalty/2
		if np.isnan(L_tot):
			pdb.set_trace()
		# W1.check_W()
		# W2.check_W()
		# print("Iter %s, Total Loss: %s " %(iteration_count, L_tot)),
		# print("Predicted Value: %s, Actual Value: %s" %(pred, train_y[list_location]))
		# print('Elapsed Time: %s' %(time.time() - start_time))
		#backprop
		dL_da2  = S1.backprop() # delta3*g'
		# delta2  = W2.W.T*dL_da2
		da2_dw2 = W2.backprop(Z1) #
		da2_dz1 = W2.W
		dz1_da1 = R1.backprop(A1)
		da1_dw1 = train_x[list_location]

		dL_dw2 += da2_dw2.T * dL_da2
		# print("shape of dL_dw2: %s" %dL_dw2.shape)
		part1 = dL_da2 * da2_dz1.T
		part2 = part1 * dz1_da1.T
		dL_dw1 += np.outer(part2, da1_dw1.T).T
		# print("shape of dL_dw1: %s, %s" %dL_dw1.shape)
		# pdb.set_trace()
		#gradient descent
	dL_dw2 = dL_dw2/mini_batch
	dL_dw1 = dL_dw1/mini_batch
	W2.backward(dL_dw2, learning_rate = alpha, momentum = 0.8, l2_penalty = l2_penalty)
	# W2.check_W()
	W1.backward(dL_dw1, learning_rate = alpha, momentum = 0.8, l2_penalty = l2_penalty)
	# W1.check_W()
	alpha = max(0.05, alpha*anneal)
	L_history.append(L_tot)
	acc_history.append(accuracy/iter_mini)
	if abs(accuracy/iter_mini - acc_prev) < 0.01:
		acc_count += 1
	else:
		acc_count = 0
	acc_prev = accuracy/iter_mini
	print('Current Accuracy: %s' %str(accuracy/iter_mini))

####### Verifying against test set  #########
accuracy = 0
for i in range(test_set_size):
	S1.set_y_data(test_y[i])
	A1 = W1.forward(test_x[i])
	Z1 = R1.forward(A1)
	A2 = W2.forward(Z1)
	L, pred = S1.forward(A2)
	if pred == test_y[i]:
		accuracy += 1.0
print('Test Accuracy: %s' %str(accuracy/test_set_size))

if PlotOn:
	plt.subplot(2,1,1)
	plt.plot(L_history, 'ro')
	plt.subplot(2,1,2)
	plt.plot(acc_history, 'bo')
	plt.show()
	plt.pause(0.001)
pdb.set_trace()
input('Press Enter to complete')
# pdb.set_trace()
