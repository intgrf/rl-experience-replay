import numpy as np
import random


class SumTree:
    data_idx = 0  # current index to write new data

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # (capacity - 1) for parent nodes and (capacity) for leafs
        self.data = np.zeros(capacity, dtype=object)

    def add(self, element, priority):
        tree_idx = self.data_idx + self.capacity - 1
        self.data[self.data_idx] = element
        self.update(tree_idx, priority)
        self.data_idx = (self.data_idx + 1) % self.capacity

    def update(self, idx, priority):
        diff = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += diff

    def _find(self, cur_idx, val):
        left = 2 * cur_idx + 1
        right = left + 1

        if left >= len(self.tree): # if cur node is leaf
            return cur_idx

        if self.tree[left] >= val:
            return self._find(left, val)
        else:
            return self._find(right, val - self.tree[left])

    def get(self, val):
        tree_idx = self._find(0, val)
        idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[idx]

    def total(self):
        return self.tree[0]


class PrioritizedMemory:
    cur_length = 0

    def __init__(self, capacity, eps=1e-5, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(capacity)
        self.eps = eps  # prevents zero priority
        self.alpha = alpha  # exponent in sampling probability
        self.beta = beta  # exponent in importance-sampling weights

    def _get_priority(self, td_error):
        #return (td_error + self.eps) ** self.alpha
        return td_error + self.eps

    def update(self, idx, td_error):
        priority = self._get_priority(td_error)
        self.sum_tree.update(idx, priority)

    def sample(self, k):
        batch, batch_idx, weights = [], [], []
        p_sum = self.sum_tree.total()
        segment = p_sum / k

        for i in range(k):
            left = segment * i
            right = segment * (i + 1)

            s = random.uniform(left, right)
            idx, priority, transition = self.sum_tree.get(s)
            priority = priority ** self.alpha

            batch_idx.append(idx)
            batch.append(transition)
            prob = priority / p_sum
            weight = (self.cur_length * prob) ** (-self.beta)
            weights.append(weight)

        max_weight = max(weights)

        for i in range(k):
            weights[i] /= max_weight
        # weights = np.power(self.cur_length * np.array(probs), -self.beta)
        # weights /= weights.max()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return batch_idx, (states, actions, rewards, next_states, dones), weights

    def add(self, transition):
        priority = 1.0 if self.cur_length == 0 else self.sum_tree.tree.max()
        self.sum_tree.add(transition, priority)
        if self.cur_length < self.sum_tree.capacity:
            self.cur_length += 1
