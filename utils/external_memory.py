import torch
import random
import numpy as np
import copy


class ExtMemory:
    def __init__(self, patterns_shape, labels_shape, size=500):
        self.size = size
        self.patterns_shape = patterns_shape
        self.labels_shape = labels_shape

        self.memory = torch.zeros((self.size, *self.patterns_shape), dtype=torch.float32)
        self.labels = torch.zeros((self.size, *self.labels_shape), dtype=torch.float32)
        self.n_batches = 0

    def _replace_patterns(self, patterns):

        idx_to_add = np.random.choice(range(self.size), size=len(patterns))
        self.memory[idx_to_add] = patterns[0]
        self.labels[idx_to_add] = patterns[1]
        # if self.n_saved_batches == 0:
        #     h = self.RMsize
        # else:
        #     h = self.RMsize // self.n_saved_batches
        #
        # self.n_saved_batches += n_batches
        #
        # add_id = np.random.choice(range(patterns.shape[0]), size=h)
        # replace_id = np.random.choice(range(self.RMsize), size=h)
        #
        # print(f'add {h} patterns to RM with size {self.RMsize}')
        #
        # self.memory['activations'][replace_id] = copy.deepcopy(patterns[add_id])
        # self.memory['labels'][replace_id] = copy.deepcopy(labels[add_id])

    def add_patterns(self, patterns):
        self.n_batches += 1

        h = min(len(patterns), self.size // self.n_batches)

        idx_to_add = np.random.choice(range(len(patterns)), size=h)

        self._replace_patterns(patterns[idx_to_add])

    def get_memory(self):
        return self.memory


if __name__ == '__main__':
    rm = ExtMemory((10, 10, 10))
    print(rm.size)
