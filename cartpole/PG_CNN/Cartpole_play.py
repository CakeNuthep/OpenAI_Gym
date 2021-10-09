import gym
from PG_CNN_test import Agent

env = gym.make('CartPole-v1')

agent = Agent(env,'model/neural_network/model.h5')
agent.play(number_of_episode=1,isRender=True,isTrain=False)

env.close()