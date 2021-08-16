import gym
import matplotlib.pyplot as plt
from ddqn_PER import Agent

env = gym.make('CartPole-v1')

agent = Agent()
agent.play(env,number_of_episode=200,isRender=False,isTrain=True)

env.close()

plt.plot(agent.reward_for_each_episode)
plt.title('Performance over time')
plt.ylabel('Total reward')
plt.xlabel('Episode')
plt.show()