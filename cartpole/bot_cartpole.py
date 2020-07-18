import gym
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
class Agent():
	def __init__(self):
		n_bins = 10
		self.cart_position_bins = pd.cut([-1.3,1.3],bins=n_bins,retbins=True)[1][1:-1]
		self.cart_velocity_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
		self.pole_angle_bins = pd.cut([-0.3,0.3],bins=n_bins,retbins=True)[1][1:-1]
		self.angle_rate_bins = pd.cut([-3,3],bins=n_bins,retbins=True)[1][1:-1]
		self.q_table = np.zeros((n_bins,n_bins,n_bins,n_bins)+(2,))
		self.learning_rate = 0.05
		self.discount_factor = 0.95
		self.epsilon = 0.5
		self.decay_factor = 0.999
		self.reward_for_each_episode = []
	def play(self, env, number_of_episode=3000, isRender = False):
		for i_episode in range(number_of_episode):
			print("Episode {} of {}".format(i_episode + 1, number_of_episode))
			observation = env.reset()
			cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
			state = (self.__to_bin(cart_position, self.cart_position_bins),
					self.__to_bin(cart_velocity,self.cart_velocity_bins),
					self.__to_bin(pole_angle,self.pole_angle_bins),
					self.__to_bin(angle_rate_of_change, self.angle_rate_bins))
			self.epsilon *= self.decay_factor
			total_reward = 0

			end_game = False
			while not end_game:
				if isRender:
					env.render()
				if self.__qTableIsEmpty(state) or self.__probability(self.epsilon):
					action = self.__getActionByRandomly(env)
				else:
					action = self.__getActionWithHighestExpectedReward(state)
				new_observation, reward, end_game, _ = env.step(action)
				if end_game:
					reward = -200
				else:
					total_reward += reward
				new_cart_position, new_cart_velocity, new_pole_angle, new_angle_rate_of_change = new_observation
				new_state = (self.__to_bin(new_cart_position,self.cart_position_bins),
						self.__to_bin(new_cart_velocity,self.cart_velocity_bins),
						self.__to_bin(new_pole_angle,self.pole_angle_bins),
						self.__to_bin(new_angle_rate_of_change,self.angle_rate_bins))
				#update q_table
				self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self._getExpectedReward(new_state) - self.q_table[state][action])
				#total_reward += reward
				state = new_state
			self.reward_for_each_episode.append(total_reward)
			#print(tabulate(self.q_table, showindex="always", headers=["State", "Action 0 (Forward 1 step)", "Action 1 (Back to 0)"]))

	def __qTableIsEmpty(self, state):
		return np.sum(self.q_table[state]) == 0

	def __probability(self, probability):
		return np.random.random() < probability

	def __getActionByRandomly(self, env):
		return env.action_space.sample()

	def __getActionWithHighestExpectedReward(self, state):
		return np.argmax(self.q_table[state])

	def _getExpectedReward(self, state):
		return np.max(self.q_table[state])

	def __to_bin(self,value,bins):
		return np.digitize(x=value,bins=bins)

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

