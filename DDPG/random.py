import numpy as np


class RandomProcess:
    def __init__(self, rand_proc_type):
        if rand_proc_type == 'normal':
            self.type = 'normal'

    def rand_explore(self, a=1, b=1):
        if self.type == 'normal':
            return 1 / 3 * np.random.randn(a, b)

