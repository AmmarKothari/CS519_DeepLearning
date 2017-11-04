
import cPickle
import numpy as np
import pdb
import matplotlib.pyplot as plt
import random
import pickle
import os
import csv


cur_path = os.getcwd()
path_files = os.listdir(cur_path)
# pdb.set_trace()
csv_files = [f for f in path_files if f.split('.')[-1] == 'txt']

plt.ion()
#### HIDDEN NODES PLOT ####
H_files = [f for f in csv_files if f.split('_')[1] == 'LR01' and f.split('.txt')[0].split('_')[-1] == str(10)]
Hidden_data = np.empty((0,4))
plt.figure()
for hf in H_files:
	hidden_count = int(hf.split('_')[0].split('H')[1])
	with open(hf) as datafile:
		cur_data = np.empty((0,4))
		datareader = csv.reader(datafile, delimiter = ' ')
		for row in datareader:
			# pdb.set_trace()
			cur_data = np.append(cur_data, np.array(row).astype(float).reshape(-1,4), axis = 0)
		# cur_data = cur_data.reshape(4,-1).T
		Hidden_data = np.append(Hidden_data, cur_data, axis = 0)
		# print(cur_data[:,2])
		plt.plot(cur_data[:,1], 'o-', label = str(hidden_count))
		plt.pause(0.01)
plt.title('Effect of Number of Hidden Node on Test Accuracy')
plt.legend()
plt.show()


#### LEARNING RATE PLOT ####
lr_files = [f for f in csv_files if f.split('_')[0] == 'H50' and f.split('.txt')[0].split('_')[-1] == str(10)]
lr_data = np.empty((0,4))
plt.figure()
for lrf in lr_files:
	lr_count = lrf.split('_')[1].split('LR')[1]
	if lr_count[0] == '0':
		lr_count = '.' + lr_count
	elif lr_count[0] == 'A':
		lr_count = '.' + lr_count[1:]
	with open(lrf) as datafile:
		cur_data = np.empty((0,4))
		datareader = csv.reader(datafile, delimiter = ' ')
		for row in datareader:
			# pdb.set_trace()
			cur_data = np.append(cur_data, np.array(row).astype(float).reshape(-1,4), axis = 0)
		# cur_data = cur_data.reshape(4,-1).T
		lr_data = np.append(lr_data, cur_data, axis = 0)
		# print(cur_data[:,2])
		plt.plot(cur_data[:,1], 'o-', label = str(lr_count))
		plt.pause(0.01)
plt.title('Effect of Learning Rate on Test Accuracy')
plt.legend()
plt.show()

#### BATCH SIZE PLOT ####
BS_files = [f for f in csv_files if f.split('_')[0] == 'H50' and f.split('_')[1] == 'LR01']
BS_data = np.empty((0,4))
plt.figure()
for BSf in BS_files:
	BS_count = int(BSf.split('_')[-1].split('.txt')[0])
	with open(BSf) as datafile:
		cur_data = np.empty((0,4))
		datareader = csv.reader(datafile, delimiter = ' ')
		for row in datareader:
			cur_data = np.append(cur_data, np.array(row).astype(float).reshape(-1,4), axis = 0)
		BS_data = np.append(BS_data, cur_data, axis = 0)
		plt.plot(cur_data[:,1], 'o-', label = str(BS_count))
		plt.pause(0.01)
plt.title('Effect of Batch Size on Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.legend()
plt.show()

pdb.set_trace()


