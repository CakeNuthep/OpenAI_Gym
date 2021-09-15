import gym
import numpy as np
import tensorflow as tf
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model,save_model
from keras.layers import InputLayer, Dense, Conv2D, Flatten
from keras import optimizers

from collections import deque
import random
from Memory import *
import cv2

class Agent():
	def __init__(self,env,path_neural_network=None,path_target_model=None):
		# self.memory = deque(maxlen=1000000)
		self.memory = Memory(1000000)  # PER Memory
		self.batch_size = 64
		self.env = env
		self.REM_STEP = 4
		self.ROWS = 160
		self.COLS = 240
		self.image_memory = np.zeros((self.ROWS, self.COLS,self.REM_STEP))
		self.state_size = (self.ROWS, self.COLS,self.REM_STEP)
		
		self.learning_rate = 0.001
		self.discount_factor = 0.95
		self.epsilon = 0.5
		self.decay_factor = 0.999
		self.TAU = 0.5
		self.reward_for_each_episode = []
		self.neural_network = NeuralNetwork(self.state_size,self.env.action_space.n,self.learning_rate)
		self.target_model = NeuralNetwork(self.state_size,self.env.action_space.n,self.learning_rate)
		

		if path_neural_network != None:
			self.neural_network.load_weights(path_neural_network)
		if path_target_model != None:
			self.target_model.load_weights(path_target_model)

		

	def play(self, number_of_episode=500, isRender = False,isTrain = True):
		max_total_reward = -200
		for i_episode in range(number_of_episode):
			#print("Episode {} of {}".format(i_episode + 1, number_of_episode))
			state = self.reset()
			# state = np.reshape(state,[1,4])
			
			total_reward = 0

			end_game = False
			while not end_game:
				if isRender:
					self.env.render()
				if self.__probability(self.epsilon):
					action = self.__getActionByRandomly()
				else:
					action = self.__getActionWithHighestExpectedReward(state)
				new_state, reward, end_game, _ = self.step(action)
				# new_state = np.reshape(new_state,[1,4])
				if end_game:
					reward = -200
				else:
					total_reward += reward
				
				self.remember(state,action,reward,new_state,end_game)
				
				state = new_state
				self.experience_replay()
			if max_total_reward <= total_reward:
				max_total_reward = total_reward
				self.neural_network.save_weights('model/neural_network/model.h5')
				self.target_model.save_weights('model/target_model/model.h5')
			self.reward_for_each_episode.append(total_reward)
			if isTrain:
				self.update_target_model()
			print("Episode {} of {}, reward:{}".format(i_episode + 1, number_of_episode,total_reward))
			print(self.memory.tree.n_entries)

	def GetImage(self):
		img = self.env.render(mode='rgb_array')
  
		img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
		img_rgb_resized[img_rgb_resized < 255] = 0
		img_rgb_resized = img_rgb_resized / 255

		self.image_memory = np.roll(self.image_memory, 1, axis = 2)
		self.image_memory[:,:,0] = img_rgb_resized

		# self.imshow(self.image_memory,0)

		return np.expand_dims(self.image_memory, axis=0)

	def reset(self):
		self.env.reset()
		for i in range(self.REM_STEP):
			state = self.GetImage()
		return state

	def step(self,action):
		next_state, reward, done, info = self.env.step(action)
		next_state = self.GetImage()
		return next_state, reward, done, info

	def __probability(self, probability):
		return np.random.random() < probability

	def __getActionByRandomly(self):
		return self.env.action_space.sample()

	def __getActionWithHighestExpectedReward(self, state):
		
		return np.argmax(self.neural_network.predict_expected_rewards_for_each_action(state))

	def _getExpectedReward(self, state):
		return np.max(self.neural_network.predict_expected_rewards_for_each_action(state))

	def remember(self,state,action,reward,next_state,done):
		# Calculate TD-Error for Prioritized Experience Replay
		td_error = reward + self.discount_factor * self.__getActionWithHighestExpectedReward(next_state) - self.__getActionWithHighestExpectedReward(state)
		# Save TD-Error into Memory
		self.memory.add(td_error,(state,action,reward,next_state,done))
		# self.memory.append((state,action,reward,next_state,done))

	def experience_replay(self):
		if self.memory.tree.n_entries < self.batch_size:
			return
		# batch = random.sample(self.memory, self.batch_size)
		batch, idxs, is_weight = self.memory.sample(self.batch_size)



		list_state = np.zeros((self.batch_size,)+self.state_size)
		list_next_state = np.zeros((self.batch_size,) + self.state_size)
		list_action, list_reward, list_done = [],[],[]
		for i in range(self.batch_size):
			list_state[i] = batch[i][0]
			list_action.append(batch[i][1])
			list_reward.append(batch[i][2])
			list_next_state[i] = batch[i][3]
			list_done.append(batch[i][4])
		q_values = self.neural_network.predict_expected_rewards_for_each_action(list_state)
		target_next = self.neural_network.predict_expected_rewards_for_each_action(list_next_state)
		target_val = self.target_model.predict_expected_rewards_for_each_action(list_next_state)

		for i in range(len(batch)):
			q_update = list_reward[i]
			if not list_done[i]:
				a = np.argmax(target_next[i])
				q_update = list_reward[i] + self.discount_factor * (target_val[i][a])
				#q_update = reward + self.discount_factor * self._getExpectedReward(state_next)
			#q_values = self.neural_network.predict_expected_rewards_for_each_action(state)
			q_values[i][list_action[i]] = q_update
		self.neural_network.train(list_state,q_values,is_weight)
		self.epsilon *= self.decay_factor

	def update_target_model(self):
		q_model_theta = self.neural_network.get_model()
		target_model_theta = self.target_model.get_model()
		counter = 0
		for q_weight, target_weight in zip(q_model_theta,target_model_theta):
			target_weight = target_weight * (1-self.TAU) + q_weight*self.TAU
			target_model_theta[counter]=target_weight
			counter += 1
		self.target_model.set_model(target_model_theta)

class NeuralNetwork(Sequential):
	def __init__(self,input_shape,action_space,learning_rate):
		super().__init__()

		# Add CNN
		self.add(Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_last"))
		self.add(Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_last"))
		self.add(Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_last"))
		self.add(Flatten())

		self.add(Dense(24,activation="relu",name="layer1"))
		self.add(Dense(24,activation="relu",name="layer2"))
		self.add(Dense(action_space,activation="linear",name="layer3"))
		self.compile(loss='mse',optimizer=optimizers.Adam(lr=learning_rate))

	def train(self,state,target_output,sample_weight=None):
		self.fit(state,target_output,epochs=1,verbose=0,sample_weight=sample_weight)

	def predict_expected_rewards_for_each_action(self,state):
		return self.predict(state)

	def get_model(self):
		return self.get_weights()

	def set_model(self,model):
		self.set_weights(model)


