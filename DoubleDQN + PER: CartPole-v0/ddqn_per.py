import torch
from torch import nn, optim
import torch.nn.functional as F
import copy
import random
from memory import PrioritizedMemory
from torch.utils.tensorboard import SummaryWriter

class DDQN:
    device = torch.device('cuda')

    min_eps = 0.1
    max_eps = 0.5

    memory_capacity = 20000

    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    update_step = 1000
    max_episode = 10000
    max_step = 50000
    optimize_step = 200

    def __init__(self, model, target_model, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.model = model
        self.target_model = target_model

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        self.memory = PrioritizedMemory(self.memory_capacity)

        self.writer = SummaryWriter('./result/cart-pole-3')

    def select_action(self, epsilon, state, exploit=False):
        if random.random() < epsilon:
            return random.randint(0, self.action_size-1)
        if exploit:
            return self.target_model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()
        else:
            return self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def update_target_model(self):
        self.target_model = copy.deepcopy(self.model)

    def exploit(self, render=True):
        state = self.env.reset()
        r = 0.
        is_terminal = False

        while not is_terminal:
            eps = 0.
            if render:
                self.env.render()
            action = self.select_action(eps, state, exploit=True)
            state, reward, is_terminal, _ = self.env.step(action)
            r += reward
        return r

    def compute_td_errors(self):
        idxs, batch, weights = self.memory.sample(self.batch_size)

        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        done = torch.tensor(done).to(self.device).float()
        weights = torch.tensor(weights).to(self.device).float()

        q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)

        target_q = self.target_model(next_state).max(1)[0]
        target_q = reward + target_q * self.gamma * (1 - done)

        loss = ((q - target_q.detach()).pow(2) * weights)
        td_error = loss.data.cpu().numpy()
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, td_error[i])

        self.optimizer.step()

    def train(self):
        episode_reward = 0
        state = self.env.reset()
        done = False
        for step in range(self.max_step):
            print(step)
            eps = self.max_eps - (self.max_eps - self.min_eps) * step / self.max_step
            action = self.select_action(eps, state)

            next_state, reward, done, _ = self.env.step(action)
            self.memory.add((state, action, reward, next_state, done))

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                self.writer.add_scalar('exploration_reward', episode_reward,
                                       global_step=step)
                episode_reward = 0

            if self.memory.cur_length > self.batch_size:
                self.compute_td_errors()

            if step % self.update_step == 0:
                self.update_target_model()
                r = self.exploit(render=False)
                self.writer.add_scalar('exploitation_reward', r,
                                       global_step=step)
