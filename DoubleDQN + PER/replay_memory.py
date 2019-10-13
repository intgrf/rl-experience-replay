import numpy as np

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
        self.update_tree(tree_idx, priority)
        self.data_idx = (self.data_idx + 1) % self.capacity

    def update_tree(self, idx, priority):
        diff = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += diff

    def _find_in_tree(self, cur_idx, val):
        left = 2 * cur_idx + 1
        right = left + 1

        if left >= len(self.tree): # if cur node is leaf
            return cur_idx

        if self.tree[left] >= val:
            return self._find_in_tree(left, val)
        else:
            return self._find_in_tree(right, val - self.tree[left])

    def get(self, val):
        tree_idx = self._find_in_tree(0, val)
        idx = tree_idx - self.capacity + 1
        return self.data[idx]

    def total(self):
        return self.tree[0]

