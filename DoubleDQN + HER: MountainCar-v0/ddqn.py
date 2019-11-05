import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import random
import copy
from memory import Memory
import torch.nn.functional as F


class DDQN:
    memory_capacity = 10**6
    batch_size = 128
    gamma = 0.98
    eps = 0.2

    goal = np.array([0.55])
    k_future_goals = 4
    n_batches = 40

    max_epoch = 15
    max_cycle = 50
    max_episode = 16
    max_step = 200

    max_eps = 0.5
    min_eps = 0.1

    device = torch.device('cuda')

    def __init__(self, model, target_model, optimizer, env):
        self.model = model
        self.target_model = target_model

        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape[0]

        self.memory = Memory(self.memory_capacity)

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optimizer

        self.writer = SummaryWriter('./result/mc/mountaincar-1e-3_128_future-no-rs-2')

    def select_action(self, epsilon, state, exploit=False):
        state = np.concatenate((state, self.goal))
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        if exploit:
            prediction = self.target_model(torch.tensor(state).to(self.device).float().unsqueeze(0))
            return prediction[0].max(0)[1].view(1, 1).item()
        else:
            prediction = self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))
            return prediction[0].max(0)[1].view(1, 1).item()

    def exploit(self, render=True):
        state = self.env.reset()
        r = 0.
        done = False
        while not done:
            eps = 0.
            if render:
                self.env.render()
            action = self.select_action(eps, state, exploit=True)
            state, reward, done, _ = self.env.step(action)
            r += reward
        return r

    def optimize(self, batch, step_num):
        state, action, reward, next_state, is_terminal = batch

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q = self.target_model(next_state).max(1)[0].detach()
        target_q = reward + target_q * self.gamma

        q = self.model(state).gather(1, action.unsqueeze(1))

        loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))
        self.writer.add_scalar('loss', loss.item(), global_step=step_num)
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_reward_and_done(self, state, goal):
        if np.array_equal(state, goal):
            return 0, True
        return -1, False

    def train(self, her_strategy='final'):
        total_cycles = 0; total_episodes = 0; total_steps = 0
        for epoch in range(self.max_epoch):
            print(epoch)

            for cycle in range(self.max_cycle):
                total_cycles += 1

                for episode in range(self.max_episode):
                    total_episodes += 1; episode_reward = 0
                    episode_history = []

                    state = self.env.reset()
                    # play episode
                    for step in range(self.max_step):
                        # select action: epsilon-greedy
                        eps = self.max_eps - (self.max_eps - self.min_eps) * total_steps / (self.max_epoch * self.max_episode * self.max_step)
                        action = self.select_action(eps, state)
                        # execute action
                        next_state, reward, done, _ = self.env.step(action)

                        episode_history.append((state, action, reward, next_state, done))
                        episode_reward += reward
                        self.memory.append((np.concatenate((state, self.goal)), action, reward,
                                            np.concatenate((next_state, self.goal)), done))
                        state = next_state
                        total_steps += 1
                        if done:
                            break
                    self.writer.add_scalar('explorer_reward', episode_reward,
                                           global_step=total_episodes)

                    # store transitions in replay memory
                    if her_strategy == 'final':
                        achieved_goal = episode_history[-1][-2][:1]   # final state of the episode -> new goal
                        for state, action, reward, next_state, done in episode_history:
                            reward, done = self.update_reward_and_done(next_state[:1], achieved_goal)

                            self.memory.append((np.concatenate((state, achieved_goal)), action,
                                                reward, np.concatenate((next_state, achieved_goal)), done))

                    if her_strategy == 'future':
                        episode_len = len(episode_history)
                        for i in range(episode_len):
                            state, action, reward, next_state, done = episode_history[i]
                            for t in range(self.k_future_goals):
                                new_goal = episode_history[random.randint(i, episode_len-1)][-2][:1]
                                reward, done = self.update_reward_and_done(next_state[:1], new_goal)
                                self.memory.append((np.concatenate((state, new_goal)), action,
                                                    reward, np.concatenate((next_state, new_goal)), done))

                # optimization step
                if len(self.memory) > self.batch_size:
                    for _ in range(self.n_batches):
                        self.optimize(list(zip(*self.memory.sample(self.batch_size))), total_cycles)

                # update target model
                self.target_model = copy.deepcopy(self.model)
                r = self.exploit(render=False)
                self.writer.add_scalar('exploit_reward', r, global_step=total_cycles)

