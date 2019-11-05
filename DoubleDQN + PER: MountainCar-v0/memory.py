import numpy as np
import random


class SumTree:
    data_idx = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.data_idx + self.capacity - 1

        self.data[self.data_idx] = data
        self.update(idx, p)

        self.data_idx += 1
        if self.data_idx >= self.capacity:
            self.data_idx = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        tree_idx = self._retrieve(0, s)
        idx = tree_idx - self.capacity + 1

        return tree_idx, self.tree[tree_idx], self.data[idx]


class PrioritizedMemory:

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.current_length = 0

    def add(self, state, action, reward, next_state, done):
        if self.current_length is 0:
            priority = 1.0
        else:
            priority = self.sum_tree.tree.max()
        self.current_length = self.current_length + 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size):
        batch_idx = []
        batch = []
        is_weights = []

        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            is_weight = (self.sum_tree.total() * prob) ** (-self.beta)
            is_weights.append(is_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return batch_idx, (state_batch, action_batch, reward_batch, next_state_batch, done_batch), is_weights

    def update_priority(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
