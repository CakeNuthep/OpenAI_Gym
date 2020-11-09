import gym
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras import optimizers

from collections import deque
import random

class Agent():
	def __init__(self):
		self.memory = deque(maxlen=1000000)
		self.batch_size = 20

		
		self.learning_rate = 0.001
		self.discount_factor = 0.95
		self.epsilon = 0.5
		self.decay_factor = 0.999
		self.TAU = 0.5
		self.reward_for_each_episode = []
		self.neural_network = NeuralNetwork(4,2,self.learning_rate)
		self.target_model = NeuralNetwork(4,2,self.learning_rate)

	def play(self, env, number_of_episode=50, isRender = False):
		for i_episode in range(number_of_episode):
			#print("Episode {} of {}".format(i_episode + 1, number_of_episode))
			state = env.reset()
			state = np.reshape(state,[1,4])
			
			total_reward = 0

			end_game = False
			while not end_game:
				if isRender:
					env.render()
				if self.__probability(self.epsilon):
					action = self.__getActionByRandomly(env)
				else:
					action = self.__getActionWithHighestExpectedReward(state)
				new_state, reward, end_game, _ = env.step(action)
				new_state = np.reshape(new_state,[1,4])
				if end_game:
					reward = -200
				else:
					total_reward += reward
				
				self.remember(state,action,reward,new_state,end_game)
				
				state = new_state
				self.experience_replay()
			self.reward_for_each_episode.append(total_reward)
			self.update_target_model()
			print("Episode {} of {}, reward:{}".format(i_episode + 1, number_of_episode,total_reward))

	

	def __probability(self, probability):
		return np.random.random() < probability

	def __getActionByRandomly(self, env):
		return env.action_space.sample()

	def __getActionWithHighestExpectedReward(self, state):
		return np.argmax(self.neural_network.predict_expected_rewards_for_each_action(state)[0])

	def _getExpectedReward(self, state):
		return np.max(self.neural_network.predict_expected_rewards_for_each_action(state)[0])

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def experience_replay(self):
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)

		state_size = 4
		list_state = np.zeros((self.batch_size,state_size))
		list_next_state = np.zeros((self.batch_size, state_size))
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
		self.neural_network.train(list_state,q_values)
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
	def __init__(self,observation_space,action_space,learning_rate):
		super().__init__()
		self.add(Dense(24,input_shape=(observation_space,),activation="relu"))
		self.add(Dense(24,activation="relu"))
		self.add(Dense(action_space,activation="linear"))
		self.compile(loss='mse',optimizer=optimizers.Adam(lr=learning_rate))

	def train(self,state,target_output):
		self.fit(state,target_output,epochs=1,verbose=0)

	def predict_expected_rewards_for_each_action(self,state):
		return self.predict(state)

	def get_model(self):
		return self.get_weights()

	def set_model(self,model):
		self.set_weights(model)
env = gym.make('CartPole-v1')

agent = Agent()

agent.play(env)
agent.play(env,number_of_episode=1,isRender=True)
env.close()
plt.plot(agent.reward_for_each_episode)

plt.title('Performance over time')

plt.ylabel('Total reward')
plt.xlabel('Episode')

plt.show()

