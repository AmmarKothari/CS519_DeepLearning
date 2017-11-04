import gym
from gym import wrappers
# from gym_recording.wrappers import TraceRecordingWrapper
import os
import pdb
import numpy as np
import csv
from RecordData import *

########################
#### RUN SETTINGS ######
########################
Render = False
Record = True

########################
####### CONSTANTS ######
########################
csv_filename = 'Cartpole_Long.csv'
iterations = 1000




### SETTING UP DATA RECORDING ####
RC = RecordData(csv_filename)

### SETTING UP ENVIORNMENT ###
env = gym.make('CartPoleHeavy-v0')
env.length = 3 # can change some variables of the enviornment through class
# env = gym.make('CartPole-v0')
# env.theta_threshold_radians = np.pi #testing out other env reset values

# env.directory = os.getcwd()
# env.monitor.start('/tmp/cartpole-experiment-1', force=True)
# cur_path = os.getcwd()
# env = wrappers.Monitor(env, cur_path, force=True)
# env = TraceRecordingWrapper(env, cur_path)


i_iter = 0

while i_iter < iterations:
	RC.csv_state_reset()
	observation = env.reset() # need to do this prior to starting a run
	for t in xrange(1000):
		if Render:
			env.render()
		action = env.action_space.sample()
		# action = action - 10
		observation, reward, done, info = env.step(action)
		RC.append(observation, action, reward)
		# print('Observation:')
		# print(observation)
		# print('Reward: %s' %reward)
		# print('Done Status: %s' %done)
		# print('Info')
		# print(info)
		if done:
			eps_str = "Episode {} finished after {} timesteps".format(i_iter, t+1)
			print(eps_str)
			i_iter += 1
					
			if Record:
				RC.write(eps_str)
				# pdb.set_trace()
				break  #need to reset env prior to next test so just breaking out of for loop







