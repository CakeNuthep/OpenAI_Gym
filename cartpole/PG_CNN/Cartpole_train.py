import gym
import matplotlib.pyplot as plt
from PG_CNN_test import Agent

env = gym.make('CartPole-v1')

agent = Agent(env)
agent.play(number_of_episode=10000,isRender=False,isTrain=True)

env.close()

plt.plot(agent.reward_for_each_episode)
plt.title('Performance over time')
plt.ylabel('Total reward')
plt.xlabel('Episode')
plt.show()