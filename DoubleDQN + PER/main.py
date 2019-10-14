from replay_memory import PrioritizedMemory
import gym
import torch
from torch import nn, optim
import copy

# define hyperparameters
MEMORY_CAPACITY = 5000
BATCH_SIZE = 128
MAX_STEP = 20000
UPDATE_STEP = 1000
MIN_EPS = 0.1
MAX_EPS = 0.5
# epsilon-greedy strategy
def select_action(epsilon, state, model):
    if random.random() < epsilon:
        return random.randint(0, 2)
    return model(torch.tensor(state).to(device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

def optimize(model, target_model, batch):
    # TODO: add model weights update
    state, action, reward, next_state, is_terminal = batch

    state = torch.tensor(state).to(device).float()
    action = torch.tensor(action).to(device)
    reward = torch.tensor(reward).to(device).float()
    next_state = torch.tensor(next_state).to(device).float()

    target_q 

def exploit(model, render = True):
    state = env.reset()
    r = 0.
    is_terminal = False

    while not is_terminal:
        eps = 0.
        if render:
            env.render()
        action = select_action(eps, state, model)
        state, reward, is_terminal, _ = env.step(action)
        r += reward
    return r

def train(model, target_model, oprimizer):
    reward_history = []
    replay_memory = PrioritizedMemory(MEMORY_CAPACITY)

    state = env.reset()
    for step in range(MAX_STEP):
        eps = MAX_EPS - (MAX_EPS - MIN_EPS) * step / MAX_STEP
        action = select_action(eps, state, model)

        # observe and store in memory
        new_state, reward, is_terminal, _ = env.step(action)
        reward = reward + 300 * (GAMMA * abs(new_state[1]) - abs(state[1]))
        # compute td-error
        # q_val =
        replay_memory.add((new_state, reward, is_terminal))

        # update state
        if is_terminal:
            state = env.reset()
            is_terminal = False
        else:
            state = new_state

        # optimization
        if replay_memory.n_transitions > BATCH_SIZE:
            indices, batch, weights = replay_memory.sample(BATCH_SIZE)
            optimize(list(zip(*batch)))

        # update target model and exploit
        if step % UPDATE_STEP:
            target_model = copy.deepcopy(model)
            r = exploit(target_model, True)
            reward_history.append(r)

     return reward_history

def init_model_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)

if __name__ == "__main__":
    device = torch.device('cuda')
    env = gym.make('MountainCar-v0')

    # create model + optimizer
    model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    target_model = copy.deepcopy(model)

    model.apply(init_model_weights)

    model.to(device)
    target_model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 1e-5)