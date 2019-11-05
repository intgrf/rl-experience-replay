from ddqn_per import DDQN

import gym
import torch
from torch import nn, optim
import copy

# cartpole-v0 parameters:
# hidden_size = 32
# batch_size = 64
# lr = 1e-3
# memory capacity = 20k
# update step = 1k
# max step = 50k

# mountaincar-v0 parameters:
# hidden_size = 256
# batch_size = 32
# lr = 6e-4
# memory capacity = 1000000
# update step = 1000
# update td error step = 4
# max step = 150k
# max eps = 0.5
# min eps = 0.1
# alpha = 0.8
# beta = 0.4

if __name__ == "__main__":
    #env = gym.make('CartPole-v0')
    env = gym.make('MountainCar-v0')
    state_size = 2
    action_size = 3
    hidden_size = 256
    model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size))

    target_model = copy.deepcopy(model)

    def init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)

    model.apply(init_weights)

    ddqn_per = DDQN(model, target_model, env, state_size, action_size)

    ddqn_per.train()