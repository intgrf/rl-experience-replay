from torch import nn, optim
from ddqn import DDQN
from memory import Memory
import copy
import gym
from bitflip import BitFlip

def init_model_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    model = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3))

    target_model = copy.deepcopy(model)
    model.apply(init_model_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    ddqn_her = DDQN(model, target_model, optimizer, env)

    ddqn_her.train(her_strategy = 'future')


