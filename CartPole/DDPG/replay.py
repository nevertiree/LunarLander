# -*- coding: utf-8 -*_

import random
from collections import deque
import numpy as np

REPLAY_BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 16


class ReplayBuffer:
    def __init__(self, max_buffer_size=REPLAY_BUFFER_SIZE, random_seed=1234):
        self.max_buffer_size = max_buffer_size
        self.current_buffer_size = 0
        random.seed(random_seed)
        # The right side of the deque contains the most recent experiences
        self.replay_buffer = deque()

    def clear(self):
        self.current_buffer_size = 0
        self.replay_buffer.clear()

    def size(self):
        return self.current_buffer_size

    def store(self, trans):
        # Transition = [State, Action, Reward, Terminal, Next_State]
        # Pop earlier transition if replay buffer is filled
        if self.current_buffer_size < self.max_buffer_size:
            self.current_buffer_size += 1
        else:
            self.replay_buffer.popleft()
        # Add newest transition into replay buffer
        self.replay_buffer.append(trans)

    def sample(self, batch_size=MINI_BATCH_SIZE):
        assert batch_size < self.max_buffer_size

        if self.current_buffer_size < batch_size:
            batch_size = self.current_buffer_size

        batch_list = random.sample(self.replay_buffer, batch_size)

        # Transition = [State, Action, Reward, Terminal, Next_State]
        batch_dict = {
            'state': np.array([x[0] for x in batch_list]),
            'action': np.array([x[1] for x in batch_list]),
            'reward': np.array([x[2] for x in batch_list]),
            'terminal': np.array([x[3] for x in batch_list]),
            'new_state': np.array([x[4] for x in batch_list])
        }

        self.clear()
        return batch_dict

    def buffer_filled(self):
        return self.current_buffer_size >= self.max_buffer_size
