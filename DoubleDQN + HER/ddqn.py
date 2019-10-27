import torch
from torch import nn, optim
import copy, random
from replay_memory import Memory
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class DDQN:
    batch_size = 128
    lr = 1e-4
    gamma = 0.95

    max_episode = 30
    update_step = 20

    goal = np.array([0.6])
    k_new_goals = 3
    memory_capacity = 1000000
    max_eps = 0.5
    min_eps = 0.1
    n_batches = 30

    def __init__(self, state_size, action_size, env):
        self.env = env
        self.action_size = action_size
        self.state_size = state_size
        self. memory = Memory(self.memory_capacity)

        self.model = nn.Sequential(
                     nn.Linear(state_size, 128),
                     nn.ReLU(),
                     nn.Linear(128, 128),
                     nn.ReLU(),
                     nn.Linear(128, action_size))

        def init_model_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

        self.model.apply(init_model_weights)

        self.target_model = copy.deepcopy(self.model)

        self.device = torch.device('cuda')
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.writer = SummaryWriter('./result/mountain-car-01')

    def exploit(self, render=True):
        state = self.env.reset()
        print('goal:', self.goal, '\n')
        r = 0.
        is_terminal = False
        i = 0
        while not is_terminal:
            i += 1
            eps = 0.
            if render:
                self.env.render()
            action = self.select_action(eps, state, exploit=True)
            state, reward, is_terminal, _ = self.env.step(action)
            r += reward
        return r

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

    def optimize(self, batch):
        state, action, reward, next_state, is_terminal = batch

        state = torch.tensor(state).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()

        target_q = torch.zeros(reward.size()[0]).float().to(self.device)
        with torch.no_grad():
            target_q[is_terminal] = self.target_model(next_state).max(1)[0].detach()[is_terminal]
        target_q = reward + target_q * self.gamma

        q = self.model(state).gather(1, action.unsqueeze(1))

        loss = nn.functional.smooth_l1_loss(q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def train(self):
        reward_history = []
        num_transitions = 0
        total_step = 0
        for epoch in range(200):
            # play episodes
            for episode in range(self.max_episode):
                total_reward = 0
                print('episode ', episode)

                # init environment
                steps_num = 0
                is_terminal = False
                states = []
                episode_history = []
                state = self.env.reset()
                for step in range(300):
                    total_step += 1
                    # select action
                    eps = self.max_eps - (self.max_eps - self.min_eps) * episode / self.max_episode
                    # print(eps)
                    #self.env.render()
                    action = self.select_action(eps, state)

                    # execute action in environment
                    next_state, reward, is_terminal, _ = self.env.step(action)
                    # store transition in episode buffer
                    episode_history.append((state, action, reward, next_state, is_terminal, self.goal))
                    self.memory.append((np.concatenate((state, self.goal)), action, reward,
                                        np.concatenate((next_state, self.goal)), is_terminal))
                    total_reward += reward

                    state = next_state
                    states.append(state)
                    steps_num += 1
                    if is_terminal:
                        break
                self.writer.add_scalar('explore_reward', total_reward, global_step=episode)
                # for all transitions
                # for i in range(steps_num):
                #     for k in range(self.k_new_goals):
                #         future = np.random.randint(i, len(episode_history))
                #         goal = episode_history[future][3][:1]
                #         state = episode_history[i][0]
                #         action = episode_history[i][1]
                #         next_state = episode_history[i][3]
                #         # print(next_state[:1])
                #         # print(goal)
                #         if next_state[0] >= goal[0]:
                #             is_terminal = True
                #         else:
                #             is_terminal = False
                #
                #         if is_terminal:
                #             reward = 0
                #         else:
                #             r   eward = -1
                #         self.memory.append((np.concatenate((state, goal)), action, reward,
                #                             np.concatenate((next_state, goal)), is_terminal))

                for i in range(steps_num):
                    state, action, reward, next_state, is_terminal, goal = episode_history[i]
                    state_goal = np.concatenate((state, goal))
                    next_state_goal = np.concatenate((next_state, goal))

                    # store transition in replay memory
                    self.memory.append((state_goal, action, reward, next_state_goal, is_terminal))

                    # store transition with achieved goal in replay memory
                    state_achieved_goal = np.concatenate((state, next_state[:1]))
                    next_state_achieved_goal = np.concatenate((next_state, next_state[:1]))
                    self.memory.append((state_achieved_goal, action, 0., next_state_achieved_goal, True))
                    num_transitions += 2

            # optimization step
            if num_transitions > self.batch_size:
                for _ in range(self.n_batches):
                    self.optimize(list(zip(*self.memory.sample(self.batch_size))))

            # update target model
            # if episode % self.update_step == 0:
            self.target_model = copy.deepcopy(self.model)
            r = self.exploit(render=True)
            reward_history.append(r)
            self.writer.add_scalar('exploit_reward', r, global_step=epoch)

        return reward_history