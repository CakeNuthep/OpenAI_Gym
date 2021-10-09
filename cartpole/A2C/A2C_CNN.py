import gym
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model,save_model
from keras.layers import InputLayer, Dense, Conv2D, Flatten
from keras import optimizers
from keras import backend as K

from collections import deque
import random
import cv2

class Agent():
	def __init__(self,env,path_neural_network=None):
		print(tf.__version__) # 1.1.0 only
		print(keras.__version__) # 2.0.5 only
		
		self.env = env
		self.REM_STEP = 4
		self.ROWS = 160
		self.COLS = 240
		self.image_memory = np.zeros((self.ROWS, self.COLS,self.REM_STEP))
		self.state_size = (self.ROWS, self.COLS,self.REM_STEP)
		
		self.learning_rate = 0.0000001
		self.states, self.actions,self.probs,self.gradients,self.rewards = [],[],[],[],[]
		self.reward_for_each_episode = []
		tf.compat.v1.disable_eager_execution()
		self.actor = NeuralNetwork(self.state_size,self.env.action_space.n)
		self.actor.add(Dense(self.env.action_space.n, activation="softmax", kernel_initializer='he_uniform'))

		self.critic = NeuralNetwork(self.state_size,self.env.action_space.n)
		self.critic.add(Dense(1, kernel_initializer='he_uniform'))
		self.critic.compile(loss='mse', optimizer=optimizers.RMSprop(lr=self.learning_rate))
		
		print("output")
		print(self.actor.output)
		print("actio_space")
		print(self.env.action_space.n)
		self.__build_train_fn(self.env.action_space.n)

		if path_neural_network != None:
			self.actor.load_weights(path_neural_network)
		

	def play(self, number_of_episode=500, isRender = False,isTrain = True):
		max_total_reward = -200
		for i_episode in range(number_of_episode):
			state = self.reset()
			
			total_reward = 0

			end_game = False
			while not end_game:
				if isRender:
					self.env.render()
				action, prob = self.act(state)
				new_state, reward, end_game, _ = self.step(action)
				
				if end_game:
					reward = -200
				else:
					total_reward += reward

				self.remember(state,action,reward)
				state = new_state

				if end_game:
					if isTrain:
						self.experience_replay()
					self.states, self.actions,self.rewards = [],[],[]
			if max_total_reward <= total_reward:
				max_total_reward = total_reward
				self.actor.save_weights('model/neural_network/model.h5')
			self.reward_for_each_episode.append(total_reward)
			print("Episode {} of {}, reward:{}".format(i_episode + 1, number_of_episode,total_reward))

	def imshow(self, image, rem_step=0):
		cv2.imshow(str(rem_step), image[...,rem_step])
		if cv2.waitKey(25) & 0xFF == ord("q"):
			cv2.destroyAllWindows()
			return

	def GetImage(self):
		img = self.env.render(mode='rgb_array')
		img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
		img_rgb_resized[img_rgb_resized < 255] = 0
		img_rgb_resized = img_rgb_resized / 255
		

		self.image_memory = np.roll(self.image_memory, 1, axis = 2)
		self.image_memory[:,:,0] = img_rgb_resized
		
		# show image frame   
        #self.imshow(self.image_memory,0)
		
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

	def act(self, state):
		action_size = self.env.action_space.n
		# Use the network to predict the next action to take, using the model
		prob = self.actor.predict(state)[0]
		action = np.random.choice(action_size, p=prob)
		return action, prob

	def remember(self,state,action,reward):
		action_size = self.env.action_space.n
		action_onehot = np.zeros([action_size] ,dtype=np.float32)
		action_onehot[action] = 1
		self.states.append(state)
		self.actions.append(action_onehot)
		self.rewards.append(reward)

	

	def discount_rewards(self, reward):
		# Compute the gamma-discounted rewards over an episode
		gamma = 0.99    # discount rate
		running_add = 0
		discounted_r = np.zeros_like(reward ,dtype=np.float32)
		for i in reversed(range(0,len(reward))):
			running_add = running_add * gamma + reward[i]
			discounted_r[i] = running_add

		discounted_r -= np.mean(discounted_r) # normalizing the result
		discounted_r /= np.std(discounted_r) # divide by standard deviation
		return discounted_r

	def train(self,states,action_onehot,discounted_reward,sample_weight=None):
		loss = self.train_fn([states, action_onehot, discounted_reward])
		print("loss")
		print(loss)

	def __build_train_fn(self,output_dim):    
		action_prob_placeholder = self.actor.output
		action_onehot_placeholder = K.placeholder(shape=(None, output_dim),
												name="action_onehot")
		discount_reward_placeholder = K.placeholder(shape=(None,),
												name="discount_reward")

		action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
		log_action_prob = K.log(action_prob)

		loss = - log_action_prob * discount_reward_placeholder
		loss = K.mean(loss)

		adam = optimizers.Adam(lr=self.learning_rate)

		updates = adam.get_updates(params=self.actor.trainable_weights,
										## constraints=[],
										loss=loss)

		print("+++input+++")
		print(self.actor.input)
		print(action_onehot_placeholder)
		print(discount_reward_placeholder)
		self.train_fn = K.function(inputs=[self.actor.input,
										action_onehot_placeholder,
										discount_reward_placeholder],
									outputs=[loss],
									updates=updates)

	def experience_replay(self):
		states = np.vstack(self.states)
		actions = np.vstack(self.actions)
		discounted_r = self.discount_rewards(self.rewards)

		# Get Critic network predictions
		self.critic.fit(states, discounted_r, epochs=1, verbose=0)

		values = self.critic.predict(states)[:, 0]
		advantages = discounted_r - values
		# advantages = discounted_r - values
		self.train(states,actions,advantages)

class NeuralNetwork(Sequential):
	def __init__(self,input_shape,action_space):
		super().__init__()
		# Add CNN
		self.add(Conv2D(64, 5, strides=(3, 3),padding="valid", input_shape=input_shape, activation="relu", data_format="channels_last"))
		self.add(Conv2D(64, 4, strides=(2, 2),padding="valid", activation="relu", data_format="channels_last"))
		self.add(Conv2D(64, 3, strides=(1, 1),padding="valid", activation="relu", data_format="channels_last"))
		self.add(Flatten())
		self.add(Dense(24,activation="relu",kernel_initializer='he_uniform',name="layer1"))
		self.add(Dense(24,activation="relu",kernel_initializer='he_uniform',name="layer2"))
		# self.add(Dense(action_space, activation="softmax", kernel_initializer='he_uniform'))


	


