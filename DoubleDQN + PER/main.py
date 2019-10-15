from ddqn import DDQN_PER
import gym
import torch
from torch import nn, optim
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')

    agent = DDQN_PER(state_size=2, action_size=3)
    rewards = agent.train(env)
    plt.plot(rewards)
