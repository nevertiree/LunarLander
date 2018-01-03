# -*- coding: utf-8 -*-

import os
import shutil

from LunarLander.DDPG.agent import Agent

import tensorflow as tf
import numpy as np


class Critic(Agent):
    def __init__(self, sess, env):
        super(Critic, self).__init__(env)

        self.sess = sess
        self.log_dir = 'log/critic'

        with tf.variable_scope("Critic"):
            self.q_value = self.critic_network()
            self.critic_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                 scope="Critic/Network")

        with tf.variable_scope("Target_Critic"):
            self.target_q_value = self.critic_network()
            self.target_critic_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                        scope="Critic/Network")

        with tf.variable_scope("Critic/Update"):

            self.loss = tf.reduce_mean(tf.squared_difference(self.q_value, self.target_q_value),
                                       name='TD-Loss')
            tf.summary.scalar(name='TD-Loss', tensor=self.loss)

            self.optimizer = tf.train.RMSPropOptimizer(0.001)
            self.train_op = self.optimizer.minimize(self.loss)

        with tf.variable_scope("Target_Critic/Update"):
            self.target_update = [tf.assign(t, self.tau * t + (1-self.tau) * e)
                                  for t, e in zip(self.target_critic_para, self.critic_para)]

        self.merge = tf.summary.merge_all()
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        self.writer = tf.summary.FileWriter(logdir=self.log_dir, graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def critic_network(self):

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            init_w = tf.random_normal_initializer(0., 1)
            init_b = tf.constant_initializer(1)

            with tf.variable_scope('layer'):
                n = 30
                weight_state = tf.get_variable('weight_state',
                                               [self.state_dim, n],
                                               initializer=init_w)
                weight_action = tf.get_variable('weight_action',
                                                [self.action_dim, n],
                                                initializer=init_w)

                bias = tf.get_variable('bias', n, initializer=init_b)

                network = tf.nn.relu(tf.matmul(self.state, weight_state) +
                                     tf.matmul(self.action, weight_action) +
                                     bias)

            with tf.variable_scope('q_value'):
                q_value = tf.layers.dense(inputs=network,
                                          units=1,
                                          kernel_initializer=init_w,
                                          bias_initializer=init_b)
        return q_value

    def get_q_value(self, network_type, s, a):
        one_hot_a = np.zeros((len(a), self.action_dim))
        for i in range(len(one_hot_a)):
            one_hot_a[i][a[i]] = 1

        if network_type == 'Estimate':
            return self.sess.run(self.q_value, feed_dict={
                self.state: s,
                self.action: one_hot_a[:np.newaxis]
            })
        else:
            return self.sess.run(self.target_q_value, feed_dict={
                self.state: s,
                self.action: one_hot_a[:np.newaxis]
            })

    def update(self, *, update_type, t_q_v=None, s=None, a=None, iter_num=None):
        # t_q_v = target_q_value ,a = action ,s = state, e_p = estimate_para

        if update_type == 'Estimate':
            a = [Critic.num_2_one_hot(a_item, self.action_dim) for a_item in a]
            summary, _ = self.sess.run([self.merge, self.train_op], feed_dict={
                self.state: s,
                self.action: a,
                self.target_q_value: t_q_v
            })
            self.writer.add_summary(summary, iter_num)
        else:
            self.sess.run(self.target_update)

