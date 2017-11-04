#! /usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import gym
import gym.wrappers


epsilon = 0.5
epsilon_decay = 0.997
epsilon_min = 0.001
gamma = 0.8
rmsprop_decay = 0.9
learning_rate = 0.05
learning_rate_decay = 0.991
n_sum_episode = 8
n_hidden_units = 128

n_max_episode = 4000


def Q_forward(model, x):
    h = np.maximum(0, np.dot(model['W1'], x) + model['B1'])
    Q = np.dot(model['W2'], h) + model['B2']
    return Q, h


def Q_backward(model, x, h, dy):
    dh = np.dot(dy, model['W2']) * (h>0)
    return {
        'B2': dy, 'W2': np.outer(dy, h),
        'B1': dh, 'W1': np.outer(dh, x)}


D = 4
H = n_hidden_units
model = {
    'B1': np.zeros((H,), dtype=np.float32),
    'W1': np.random.randn(H, D).astype(np.float32) / np.sqrt(D),
    'B2': np.zeros((2,), dtype=np.float32),
    'W2': np.random.randn(2, H).astype(np.float32) / np.sqrt(H),
}
gradient = {k: np.zeros_like(v) for k, v in model.items()}
gradient['n_step'] = 0
rmsprop = {k: np.zeros_like(v) for k, v in model.items()}


# env = gym.make('CartPole-v0')
env = gym.make('CartPoleHeavy-v0')
# env = gym.wrappers.Monitor(env, 'CartPole-v0')


for n_episode in range(n_max_episode):
    totalreward = 0
    observation = env.reset()
    xx = observation.astype(np.float32)
    qq, hh = Q_forward(model, xx)
    for _ in range(400):
        x, q, h = xx, qq, hh

        action = np.random.choice(2) if np.random.uniform() < epsilon else np.argmax(q)

        observation, reward, done, info = env.step(action)
        totalreward += reward

        dy = np.zeros_like(q)
        if done:
            dy[action] = - 1 - q[action]
        else:
            xx = observation.astype(np.float32)
            qq, hh = Q_forward(model, xx)
            dy[action] = reward + gamma * np.max(qq) - q[action]

        for k, v in Q_backward(model, x, h, dy).items():
            gradient[k] += v
        gradient['n_step'] += 1

        if done:
            print('Total Reward %s' %totalreward)
            break

    epsilon = max(epsilon_min, epsilon_decay*epsilon)

    if (n_episode % n_sum_episode) == 0:
        for k in model.keys():
            g = gradient[k] / gradient['n_step']
            rmsprop[k] *= rmsprop_decay
            rmsprop[k] += (1 - rmsprop_decay) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop[k]) + 1e-6)
            gradient[k][:] = 0
        gradient['n_step'] = 0
        learning_rate *= learning_rate_decay

env.close()