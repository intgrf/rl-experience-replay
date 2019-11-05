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
    max_beta = 0.6
    min_beta = 0.1
    memory_capacity = 1000000

    gamma = 0.99
    lr = 6e-4
    batch_size = 32
    update_step = 1000
    max_episode = 10000
    max_step = 100001
    optimize_step = 200

    def __init__(self, model, target_model, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.model = model
        self.target_model = target_model

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.memory = PrioritizedMemory(self.memory_capacity, alpha=0.6, beta=0.4)
        self.writer = SummaryWriter('./result/mc/mountain-car-11')

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
        weights = torch.tensor(weights).to(self.device).float()

        q = self.model(state).gather(1, action.unsqueeze(1))
        q = q.squeeze(1)

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q[done] = self.target_model(next_state).max(1)[0].detach()[done]
        target_q = reward.squeeze(1) + target_q * self.gamma

        td_errors = torch.pow(q - target_q, 2) * weights

        return td_errors, idxs

    def update(self):
        td_errors, idxs = self.compute_td_errors()

        loss = td_errors.mean()
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        for idx, td_error in zip(idxs, td_errors.cpu().detach().numpy()):
            self.memory.update_priority(idx, td_error)

    def optimize(self):
        batch = self.memory.sample(self.batch_size)
        batch = list(zip(*batch))
        state, action, reward, next_state, done = batch

        state = torch.tensor(state).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        action = torch.tensor(action).to(self.device)

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)

        with torch.no_grad():
            target_q[done] = self.target_model(next_state).max(1)[0].detach()[done]
        target_q = reward + target_q * self.gamma

        q = self.model(state).gather(1, action.unsqueeze(1))

        loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self):
        episode_reward = 0
        state = self.env.reset()
        done = False
        for step in range(self.max_step):
            print(step)
            eps = self.max_eps - (self.max_eps - self.min_eps) * step / self.max_step
            #eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(-0.01 * step)
            action = self.select_action(eps, state)

            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward + 300 * (self.gamma * abs(next_state[1]) - abs(state[1])), next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                self.writer.add_scalar('exploration_reward', episode_reward,
                                       global_step=step)
                episode_reward = 0

            if step % 4 == 0:
                if len(self.memory) > self.batch_size:
                    self.update()


            if step % self.update_step == 0:
                self.update_target_model()
                r = self.exploit(render=False)
                self.writer.add_scalar('exploitation_reward', r,
                                       global_step=step)
