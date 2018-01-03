# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class Agent:
    def __init__(self, env):

        self.tau = 0.99

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        with tf.variable_scope("Input"):
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state')
            self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
            self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action')
            self.q_value = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='q_value')

    @staticmethod
    def num_2_one_hot(value, max_value):
        one_hot_array = np.zeros(max_value)
        one_hot_array[value] = 1
        return one_hot_array
