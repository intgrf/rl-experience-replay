import torch
from torch import nn, optim
import torch.nn.functional as F
import copy
import random
from replay_memory import PrioritizedMemory
class DDQN_PER:
    device = torch.device('cuda')

    min_eps = 0.1
    max_eps = 0.5

    memory_capacity = 5000

    gamma = 0.99
    lr = 1e-3
    batch_size = 128
    update_step = 1000
    max_step = 50001

    def __init__(self, state_size, action_size):
        self.model = nn.Sequential(
                     nn.Linear(state_size, 32),
                     nn.ReLU(),
                     nn.Linear(32, 32),
                     nn.ReLU(),
                     nn.Linear(32, action_size))
        self.state_size = state_size
        self.action_size = action_size

        self.target_model = copy.deepcopy(self.model)

        self.model.apply(self._init_weights)
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        self.memory = PrioritizedMemory(self.memory_capacity)

    def _init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight)

    def select_action(self, epsilon, state, exploit=False):
        if random.random() < epsilon:
            return random.randint(0, self.action_size-1)
        if exploit:
            return self.target_model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()
        else:
            return self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def exploit(self, env, render=True):
        state = env.reset()
        r = 0.
        is_terminal = False

        while not is_terminal:
            eps = 0.
            if render:
                env.render()
            action = self.select_action(eps, state, exploit=True)
            state, reward, is_terminal, _ = env.step(action)
            r += reward
        return r

    def optimize(self):
        idx, batch, weights = self.memory.sample(self.batch_size)
        batch = list(zip(*batch))
        state, action, reward, next_state, is_terminal = batch

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q[is_terminal] = self.target_model(next_state).max(1)[0].detach()[is_terminal]
        target_q = (reward + target_q * self.gamma).unsqueeze(1)
        q = self.model(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            td_error = torch.abs(q - target_q)
            td_error = td_error.data
            #print(td_error)
            td_error = td_error.to('cpu', torch.float)
            td_error = td_error.numpy()
            for i in range(self.batch_size):
                self.memory.update(idx[i], td_error[i])

        loss = (torch.tensor(weights).float().to(self.device) * F.mse_loss(q, target_q)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env):
        reward_history = []

        state = env.reset()

        for step in range(self.max_step):
            print('step = ', step)
            # select action
            eps = self.max_eps - (self.max_eps - self.min_eps) * step / self.max_step
            action = self.select_action(eps, state)

            # execute action
            next_state, reward, is_terminal, _ = env.step(action)

            # reward shaping
            reward = reward + 300 * (self.gamma * abs(next_state[1]) - abs(state[1]))

            # compute td-error
            q = self.model(torch.tensor(state).to(self.device).float()).data[action].detach()
            target_q = self.target_model(torch.tensor(next_state).float().to(self.device)).data.detach()
            if is_terminal:
                y = reward
            else:
                y = reward + self.gamma * torch.max(target_q.data)
            td_error = abs(y - q)
            # store transition + td_error in memory
            self.memory.add((state, action, reward, next_state, is_terminal), td_error)

            # update weights and priorities
            if step > self.batch_size:
                self.optimize()

            if step % self.update_step == 0:
                self.update_target_model()
                r = self.exploit(env)
                reward_history.append(r)
                print('reward = ', r)

        return reward_history