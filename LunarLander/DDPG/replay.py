# -*- coding: utf-8 -*_

import random
import numpy as np
REPLAY_BUFFER_SIZE = 100
MINI_BATCH_SIZE = 10


class ReplayBuffer:
    def __init__(self, max_buffer_size=REPLAY_BUFFER_SIZE):
        self.max_buffer_size = max_buffer_size
        self.replay_buffer = []
        self.current_buffer_size = 0

    def clear(self):
        self.replay_buffer = []
        self.current_buffer_size = 0

    def store(self, trans):
        if len(self.replay_buffer) < self.max_buffer_size:
            self.replay_buffer.append(trans)
            self.current_buffer_size += 1

    def sample(self, batch_size=MINI_BATCH_SIZE):
        assert batch_size < self.max_buffer_size
        # todo min_value
        batch_list = ReplayBuffer.sub_list(raw_list=self.replay_buffer,
                                           num=batch_size)
        batch_dict = ReplayBuffer.list_2_dict(batch_list)
        self.clear()
        return batch_dict

    @staticmethod
    def sub_list(raw_list, num):
        if num > len(raw_list):
            num = len(raw_list)
        return random.sample(raw_list, num)

    @staticmethod
    def list_2_dict(raw_list):
        return {
            'state': np.array([x[0] for x in raw_list]),
            'action': np.array([x[1] for x in raw_list]),
            'reward': np.array([x[2] for x in raw_list]),
            'terminal': np.array([x[3] for x in raw_list]),
            'new_state': np.array([x[4] for x in raw_list])
        }

    def buffer_filled(self):
        return self.current_buffer_size >= self.max_buffer_size


