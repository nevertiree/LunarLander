# -*- coding: utf-8 -*-
import random
from CartPole.DDPG.agent import Agent

import tensorflow as tf
import numpy as np


class Actor(Agent):
    def __init__(self, sess, env, learning_rate=0.001):
        super(Actor, self).__init__(env)
        self.sess = sess

        # Estimate Actor Network
        with tf.variable_scope("Actor"):
            self.action = self.actor_network()
            self.actor_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='Actor/Network')
        # Update Estimate Actor Network
        with tf.variable_scope("Actor/Update"):
            # todo: Calculate Action gradient here or get it outside
            '''Action Gradient: dQ/da'''
            self.action_gradient = tf.gradients(ys=self.estimate_q_value, xs=self.action)
            self.policy_gradient = tf.gradients(ys=self.action, xs=self.actor_para,
                                                grad_ys=self.action_gradient,
                                                name='Policy_Gradient')
            self.optimizer = tf.train.AdamOptimizer(-learning_rate)
            # todo: Represent Policy Gradients
            self.estimate_update = self.optimizer.apply_gradients(zip(self.policy_gradient, self.actor_para),
                                                                  name='Estimate_Actor_Update')

        # Target Actor Network
        with tf.variable_scope("Target_Actor"):
            self.target_action = self.actor_network()
            # todo Copy value for estimate network
            self.target_actor_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                       scope='Actor/Network')

        # Update Target Actor Network
        with tf.variable_scope("Target_Actor/Update"):
            self.target_update = [tf.assign(t, self.tau * t + (1-self.tau) * e)
                                  for t, e in zip(self.target_actor_para, self.actor_para)]

        # Initialize TensorFlow Session
        self.sess.run(tf.global_variables_initializer())

    def actor_network(self):
        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
            init_w = tf.random_normal_initializer(0, 0.1)
            init_b = tf.constant_initializer(0.1)

            layer = tf.layers.dense(inputs=self.state,
                                    units=8,
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
        state = state[:np.newaxis]

        if network_type == 'Estimate':
            action_prob_list = self.sess.run(self.action, feed_dict={self.state: state})
        else:
            action_prob_list = self.sess.run(self.target_action, feed_dict={self.state: state})

        # Exploration
        action_list = []
        for action_prob in action_prob_list:
            if random.uniform(0, 1) < 0.95:
                action_list.append(np.argmax(action_prob))
            else:
                action_list.append(1-np.argmax(action_prob))

        return [int(item) for item in action_list]

    def update(self, *, update_type, state=None, action=None, estimate_q_value=None):
        if update_type == "Estimate":
            action = [Actor.num_2_one_hot(a_item, self.action_dim) for a_item in action]
            '''Update Estimate Actor Network with Policy Gradient'''
            self.sess.run(self.estimate_update, feed_dict={
                # Action Gradient == dQ/da
                # da/dθ
                # a = μ(s|θ)
                self.estimate_q_value: estimate_q_value,
                self.action: action,
                self.state: state,
            })
        else:
            self.sess.run(self.target_update)
