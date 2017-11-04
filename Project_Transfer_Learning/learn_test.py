import gym
from gym import wrappers
# from gym_recording.wrappers import TraceRecordingWrapper
import os
import pdb
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Dense, Activation

from RecordData import *
import matplotlib.pyplot as plt

########################
#### RUN SETTINGS ######
########################
Render = False
Record = False

########################
####### CONSTANTS ######
########################
csv_filename = 'Cartpole_Learn.csv'
iterations = 1000
max_t = 1000
actions_iter = 10




### SETTING UP DATA RECORDING ####
RC = RecordData(csv_filename)

### SETTING UP ENVIORNMENT ###
env = gym.make('CartPoleHeavy-v0')
# env = gym.make('CartPole-v0')
env.length = 1 # can change some variables of the enviornment through class
env.goal_scaling = 100
# env = gym.make('CartPole-v0')
# env.theta_threshold_radians = np.pi #testing out other env reset values


###### CREATING NETWORK ######
model = Sequential()
model.add(Dense(output_dim=10, input_dim=5)) #input state action, output_dim hidden units
model.add(Activation("relu"))
model.add(Dense(output_dim=1)) # output Q value
# model.add(Activation("softmax"))
# model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='mse')

# run forward through the model for one iteration (until env reset)
# at each iteration test out states with 10 different randomly sampled actions
# choose action with the highest reward
# sum up reward over run and divide by number of time steps (max of 1, min of small number)
# assume classification of 1
# back propogate with reward through network


i_iter = 0
plt.ion()
reward_plot = list()
reward_acc = list()
while i_iter < iterations:
	RC.csv_state_reset()
	observation = env.reset() # need to do this prior to starting a run
	input_record = list()
	y_record = list()
	for t in xrange(1000):
		action_list = list()
		observation_list = list()
		reward_act = list()
		reward_pred = list()
		if Render:
			env.render()
		# try for a random sampling of values and see which is best
		max_r = 0
		r_it = 0
		# for r_it in xrange(10):
		while max_r < 22 and r_it < 100:
			action = env.action_space.sample()
			inputs = np.append(observation, action).reshape(1,-1)
			r_pred = model.predict(inputs)
			r_it += 1
			max_r = max(max_r, r_pred)
			reward_pred.append(r_pred)
			action_list.append(action)

		# print('Max Reward Predicted: %s' %max_r)
		
		# choose best action
		r_ind_best = np.argmax(reward_pred)
		action_best = action_list[r_ind_best]
		# action_best = env.action_space.sample()
		# take step
		observation_new, reward, done, info = env.step(action_best)
		# train model
		inputs = np.append(observation, action_best).reshape(1,-1)
		# pdb.set_trace()
		reward_arr = r_pred
		reward_arr[0][0] = reward #scaling to match max of env
		reward_acc.append(reward_pred[r_ind_best][0][0] - reward)
		# pdb.set_trace()
		input_record.append(inputs[0])
		y_record.append(reward_arr[0][0])
		# model.fit(inputs, reward_arr, nb_epoch = 1, verbose = 0)

		RC.append(observation, action, reward)
		observation = observation_new
		# print('Observation:')
		# print(observation)
		# print('Reward: %s' %reward)
		# print('Done Status: %s' %done)
		# print('Info')
		# print(info)
		if done:
			eps_str = "Episode {} finished after {} timesteps with avg reward {}".format(i_iter, t+1, np.mean(y_record))
			print(eps_str)
			#only train at the end of each episode
			input_arr = np.array(input_record).reshape(-1,5)
			y_input = np.ones(np.array(y_record).shape) * np.mean(y_record) #give average award for all state actions
			model.fit(input_arr, y_input, nb_epoch = 1, verbose = 0)
			# pdb.set_trace()
			i_iter += 1
			# reward_plot.append(np.mean(y_input))
			# # pdb.set_trace()
			plt.plot(reward_acc, 'ro')
			plt.show()
			plt.pause(0.001)
			

			if Record:
				RC.write(eps_str)
			break  #need to reset env prior to next test so just breaking out of for loop







