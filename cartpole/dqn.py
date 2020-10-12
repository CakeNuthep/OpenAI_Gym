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

		#n_bins = 10
		#self.cart_position_bins = pd.cut([-1.3,1.3],bins=n_bins,retbins=True)[1][1:-1]
		#self.cart_velocity_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
		#self.pole_angle_bins = pd.cut([-0.3,0.3],bins=n_bins,retbins=True)[1][1:-1]
		#self.angle_rate_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
		#self.q_table = np.zeros((n_bins,n_bins,n_bins,n_bins)+(2,))
		self.learning_rate = 0.001
		self.discount_factor = 0.95
		self.epsilon = 0.5
		self.decay_factor = 0.999
		self.reward_for_each_episode = []
		self.neural_network = NeuralNetwork(4,2,self.learning_rate)

	def play(self, env, number_of_episode=50, isRender = False):
		for i_episode in range(number_of_episode):
			print("Episode {} of {}".format(i_episode + 1, number_of_episode))
			state = env.reset()
			state = np.reshape(state,[1,4])
			#cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
			#state = (self.__to_bin(cart_position, self.cart_position_bins),
			#		self.__to_bin(cart_velocity,self.cart_velocity_bins),
			#		self.__to_bin(pole_angle,self.pole_angle_bins),
			#		self.__to_bin(angle_rate_of_change, self.angle_rate_bins))
			#self.epsilon *= self.decay_factor
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
				#new_cart_position, new_cart_velocity, new_pole_angle, new_angle_rate_of_change = new_observation
				#new_state = (self.__to_bin(new_cart_position,self.cart_position_bins),
				#		self.__to_bin(new_cart_velocity,self.cart_velocity_bins),
				#		self.__to_bin(new_pole_angle,self.pole_angle_bins),
				#		self.__to_bin(new_angle_rate_of_change,self.angle_rate_bins))
				#update q_table
				self.remember(state,action,reward,new_state,end_game)
				#self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self._getExpectedReward(new_state) - self.q_table[state][action])
				#total_reward += reward
				state = new_state
				self.experience_replay()
			self.reward_for_each_episode.append(total_reward)
			#print(tabulate(self.q_table, showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

	#def __qTableIsEmpty(self, state):
	#	return np.sum(self.q_table[state]) == 0

	def __probability(self, probability):
		return np.random.random() < probability

	def __getActionByRandomly(self, env):
		return env.action_space.sample()

	def __getActionWithHighestExpectedReward(self, state):
		return np.argmax(self.neural_network.predict_expected_rewards_for_each_action(state)[0])

	def _getExpectedReward(self, state):
		return np.max(self.neural_network.predict_expected_rewards_for_each_action(state)[0])

	#def __to_bin(self,value,bins):
	#	return np.digitize(x=value,bins=bins)

	def remember(self,state,action,reward,next_state,done):
		self.memory.append((state,action,reward,next_state,done))

	def experience_replay(self):
		if len(self.memory) < self.batch_size:
			return
		batch = random.sample(self.memory, self.batch_size)
		for state,action,reward,state_next,terminal in batch:
			q_update = reward
			if not terminal:
				q_update = reward + self.discount_factor * self._getExpectedReward(state_next)
			q_values = self.neural_network.predict_expected_rewards_for_each_action(state)
			q_values[0][action] = q_update
			self.neural_network.train(state,q_values)
		self.epsilon *= self.decay_factor


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

