import numpy as np
import random

# TODO
# implement sumtree
#   1. add element
#   2. get element
#   3. update tree


class SumTree:
    data_idx = 0 # current index to write new data

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # (capacity - 1) for parent nodes and (capacity) for leafs
        self.data = np.zeros(capacity)

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

    n_transitions = 0

    def __init__(self, capacity, eps=0.01, alpha=0.6, beta=0.4):
        self.memory = SumTree(capacity)
        self.eps = eps          # prevents zero priority
        self.alpha = alpha      # exponent in sampling probability
        self.beta = beta        # exponent in importance-sampling weights

    def _get_priority(self, td_error):
        return (td_error + self.eps) ** self.alpha

    def update(self, idx, td_error):
        priority = self._get_priority(td_error)
        self.memory.update(idx, priority)

    def sample(self, k):
        batch = np.array((1, k), dtype=object)
        indices = np.array((1, k))
        probs = np.array((1, k))
        segment = self.memory.total() / k

        for i in range(k):
            left = segment * i
            right = segment * (i + 1)

            s = random.uniform(left, right)
            idx, priority, transition = self.memory.get(s)
            indices[i] = idx
            probs[i] = priority / self.memory.total()
            batch[i] = transition

        weights = np.power(self.n_transitions * probs, -self.beta)
        weights /= weights.max()

        return indices, batch, weights


    def add(self, transition, td_error):
        priority = self._get_priority(td_error)
        self.memory.add(transition, priority)
        if self.n_transitions < self.memory.capacity:
            self.n_transitions += 1