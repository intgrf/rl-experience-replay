from ddqn import DDQN
from replay_memory import Memory
import gym
import matplotlib.pyplot as plt
from bit_flip import BitFlip

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
#     n = 30
    #env = BitFlip(n, 'sparse')
    ddqn_her = DDQN(state_size=3, action_size=3, env=env)
    rewards = ddqn_her.train()

    plt.plot(rewards)
    plt.show()

