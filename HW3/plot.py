
import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np

filenames_model = ['Original.h5', 'Part1.h5', 'Part2.h5', 'Part3.h5', 'Part4_1.h5', 'Part4_2.h5']
filenames_data = ['Original.csv', 'Part1.csv', 'Part2.csv', 'Part3.csv', 'Part4_1.csv', 'Part4_2.csv']

i_iter = 0
for i_iter in range(len(filenames_data)):
# for i_iter in [2]:
	# Read in values
	results_dict = dict()
	results_orig_dict = dict()
	with open(filenames_data[i_iter]) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		for row in csv_reader:
			# dicts_from_file.append()
			# print(row)
			results_dict[row[0]] = []
			# pdb.set_trace()
			for items in row[1].strip('[]').split(','):
				results_dict[row[0]].append(float(items))

	with open(filenames_data[0]) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter = ',')
		for row in csv_reader:
			# dicts_from_file.append()
			# print(row)
			results_orig_dict[row[0]] = []
			# pdb.set_trace()
			for items in row[1].strip('[]').split(','):
				results_orig_dict[row[0]].append(float(items))

	# print(results_dict)

	# x-axis is epoch number
	# y-axis is loss/error
	x_axis = range(len(results_dict['loss']))
	x_axis_orig = range(len(results_orig_dict['acc']))
	#figure 1
	# training loss
	# validation loss
	fig1 = plt.figure()
	plt.plot(x_axis, results_dict['loss'], 'ro-', label = 'Training')
	plt.plot(x_axis, results_dict['val_loss'], 'bo-', label = 'Validation')
	if i_iter != 0:
		plt.plot(x_axis_orig, results_orig_dict['loss'], 'rs-', alpha = 0.5, label = 'Original Training')
		plt.plot(x_axis_orig, results_orig_dict['val_loss'], 'bs-', alpha = 0.5, label = 'Original Validation')
	plt.plot()
	plt.legend(loc='best')
	plt.title('Comparison of Loss for Network')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	fig1.savefig(filenames_model[i_iter].split('.')[0] + '_loss.jpg')

	# figure 2
	# training error
	# validation error
	if i_iter < 10:
		fig2 = plt.figure()
		plt.plot(x_axis, results_dict['acc'], 'ro-', label = 'Training')
		plt.plot(x_axis, results_dict['val_acc'], 'bo-', label = 'Validation')
		if i_iter != 0:
			plt.plot(x_axis_orig, results_orig_dict['acc'], 'rs-', alpha = 0.5, label = 'Original Training')
			plt.plot(x_axis_orig, results_orig_dict['val_acc'], 'bs-', alpha = 0.5, label = 'Original Validation')
		plt.legend(loc='best')
		plt.title('Comparison of Accuracy for Network')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		fig2.savefig(filenames_model[i_iter].split('.')[0] + '_error.jpg')

	print('%s: Training Loss: %.3f, Testing Loss: %.3f, Training Accuracy: %.3f, Testing Accuracy: %.3f' %(filenames_data[i_iter].split('.')[0], results_dict['loss'][-1], results_dict['val_loss'][-1], results_dict['acc'][-1], results_dict['val_acc'][-1]))

plt.show()

# plot