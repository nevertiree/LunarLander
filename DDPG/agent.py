# -*- coding: utf-8 -*-
import tensorflow as tf


class Agent:
    def __init__(self, network_type, env):
        assert network_type in ['Estimate', 'Target']
        self.network_type = network_type

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='state')
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='reward')
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='action')
