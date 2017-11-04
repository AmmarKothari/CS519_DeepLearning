
import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np

filenames_data = ['Cartpole_long_results.csv']

i_iter = 0
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

# print(results_dict)

# x-axis is epoch number
# y-axis is loss/error
x_axis = range(len(results_dict['loss']))
#figure 1
# training loss
# validation loss
fig1 = plt.figure()
plt.plot(x_axis, results_dict['loss'], 'ro-', label = 'Loss')
plt.plot()
plt.legend(loc='best')
plt.title('Comparison of Loss for Network')
plt.xlabel('Epoch')
plt.ylabel('Loss')
fig1.savefig(filenames_data[i_iter].split('.')[0] + '.jpg')
fig2 = plt.figure()
plt.plot(x_axis, results_dict['acc'], 'ro-', label = 'Accuracy')

# print('%s: Training Loss: %.3f, Testing Loss: %.3f, Training Accuracy: %.3f, Testing Accuracy: %.3f' %(filenames_data[i_iter].split('.')[0], results_dict['loss'][-1], results_dict['val_loss'][-1], results_dict['acc'][-1], results_dict['val_acc'][-1]))

plt.show()

# plot