# -*- coding: utf-8 -*-
from DDPG.agent import Agent

import tensorflow as tf
import numpy as np


class Critic(Agent):
    def __init__(self, network_type, env):
        self.agent_scope = network_type + '-Critic'
        with tf.variable_scope(self.agent_scope):
            super(Critic, self).__init__(network_type, env)
            self.loss = None
            self.optimizer = tf.train.AdamOptimizer()
            self.q_value = self.critic_network()
            self.network_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.agent_scope)

    def critic_network(self):
        scope = self.network_type
        if scope == 'Estimate':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('layer'):
                n = 30
                weight_state = tf.get_variable('weight_state',
                                               [self.state_dim, n],
                                               initializer=init_w,
                                               trainable=trainable)
                weight_action = tf.get_variable('weight_action',
                                                [self.action_dim, n],
                                                initializer=init_w, trainable=trainable)
                bias = tf.get_variable('bias', n, initializer=init_b, trainable=trainable)

                network = tf.nn.relu(tf.matmul(self.state, weight_state) +
                                     tf.matmul(self.action, weight_action) +
                                     bias)

            with tf.variable_scope('q_value'):
                q_value = tf.layers.dense(inputs=network,
                                          units=1,
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b,
                                          trainable=trainable)
        return q_value

    def get_q_value(self, sess, s, a):
        one_hot_a = np.zeros((len(a), self.action_dim))
        for i in range(len(one_hot_a)):
            one_hot_a[i][a[i]] = 1
        return sess.run(self.q_value, feed_dict={
            self.state: s,
            self.action: one_hot_a[:np.newaxis]
        })

    def update(self, *, sess, t_q_v=None, s=None, a=None, e_p=None, tau=0.99):
        # t_q_v = target_q_value ,a = action ,s = state, e_p = estimate_para

        with tf.variable_scope(self.agent_scope + '/Update'):
            # Update Estimate Network Parameter
            if self.network_type == 'Estimate':
                # Calculate the Loss Function
                estimate_q_value = self.critic_network()
                target_q_value = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='target_q_value')
                self.loss = tf.reduce_mean(tf.squared_difference(estimate_q_value, target_q_value), name='TD-Loss')

                one_hot_a = np.zeros((len(a), self.action_dim))
                for i in range(len(one_hot_a)):
                    one_hot_a[i][a[i]] = 1

                # Optimizer
                train_op = self.optimizer.minimize(self.loss)
                sess.run(tf.global_variables_initializer())
                sess.run(train_op, feed_dict={
                    self.state: s,
                    self.action: one_hot_a[:np.newaxis],
                    target_q_value: t_q_v
                })

            # Update Target Network Parameter
            else:
                sess.run([tf.assign(t, tau * t + (1-tau) * e)
                          for t, e in zip(self.network_para, e_p)])
