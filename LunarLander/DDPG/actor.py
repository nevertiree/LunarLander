# -*- coding: utf-8 -*-
import random
from LunarLander.DDPG.agent import Agent

import tensorflow as tf
import numpy as np


class Actor(Agent):
    def __init__(self, sess, env):
        super(Actor, self).__init__(env)
        self.sess = sess

        with tf.variable_scope("Actor"):
            self.action = self.actor_network()
            self.actor_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='Actor/Network')

        with tf.variable_scope("Actor/Update"):
            self.optimizer = tf.train.RMSPropOptimizer(0.001)
            self.action_gradient = tf.gradients(ys=self.q_value, xs=self.action)
            self.policy_gradient = tf.gradients(ys=self.action, xs=self.actor_para,
                                                grad_ys=self.action_gradient,
                                                name='Policy_Gradient')
            self.estimate_update = self.optimizer.apply_gradients(zip(self.policy_gradient, self.actor_para),
                                                                  name='Estimate_Actor_Update')

        with tf.variable_scope("Target_Actor"):
            self.target_action = self.actor_network()
            self.target_actor_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                       scope='Actor/Network')

        with tf.variable_scope("Target_Actor/Update"):
            self.target_update = [tf.assign(t, self.tau * t + (1-self.tau) * e)
                                  for t, e in zip(self.target_actor_para, self.actor_para)]

        self.sess.run(tf.global_variables_initializer())

    def actor_network(self):
        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            init_w = tf.truncated_normal_initializer(0., 1)
            init_b = tf.constant_initializer(0.1)

            layer = tf.layers.dense(inputs=self.state,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    name='layer')
            action = tf.layers.dense(inputs=layer,
                                     units=self.action_dim,
                                     activation=tf.nn.softmax,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     name='action')
        return action

    def choose_action(self, network_type, state):
        # Get the first element from the tensor
        if network_type == 'Estimate':
            action_prob_list = self.sess.run(self.action,
                                             feed_dict={self.state: state[:np.newaxis]})
        else:
            action_prob_list = self.sess.run(self.target_action,
                                             feed_dict={self.state: state[:np.newaxis]})

        if random.uniform(0, 1) < 0.95:
            action_list = [np.argmax(action_prob) for action_prob in action_prob_list]
        else:
            action_list = [np.random.choice(range(action_prob.shape[0]), p=action_prob.ravel())
                           for action_prob in action_prob_list]

        return action_list

    def update(self, *, update_type, s=None, a=None, e_q_v=None):
        if update_type == "Estimate":
            a = [Actor.num_2_one_hot(a_item, self.action_dim) for a_item in a]
            self.sess.run(self.estimate_update, feed_dict={
                self.state: s,
                self.action: a,
                self.q_value: e_q_v
            })
        else:
            self.sess.run(self.target_update)
