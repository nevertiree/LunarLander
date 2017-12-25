# -*- coding: utf-8 -*-
from ddpg.agent import Agent

import tensorflow as tf
import numpy as np


class Actor(Agent):
    def __init__(self, network_type, env):
        with tf.variable_scope('Actor'):
            super(Actor, self).__init__(network_type, env)
            self.state = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='State')
            self.action = self.actor_network()

            self.optimizer = tf.train.AdamOptimizer()

            if self.network_type == 'Estimate':
                self.network_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Estimate')
            else:
                self.network_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/Target')

    def actor_network(self):
        scope = self.network_type
        if scope == 'Estimate':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)

            layer = tf.layers.dense(inputs=self.state,
                                    units=32,
                                    activation=tf.nn.relu,
                                    kernel_initializer=init_w,
                                    bias_initializer=init_b,
                                    trainable=trainable,
                                    name='layer')
            action = tf.layers.dense(inputs=layer,
                                     units=self.action_dim,
                                     activation=tf.nn.softmax,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     trainable=trainable,
                                     name='action')
        return action

    def choose_action(self, sess, s):
        sess.run(tf.global_variables_initializer())
        # Get the first element from the tensor
        action_probs = sess.run(self.action, feed_dict={self.state: s[:np.newaxis]})
        action_list = [np.random.choice(range(action_prob.shape[0]), p=action_prob.ravel())
                       for action_prob in action_probs]
        return action_list

    def update(self, *, sess, s=None, action_grad=None, e_p=None, tau=0.9):
        # Update Estimate Network Parameter
        if self.network_type == 'Estimate':
            policy_grad = tf.gradients(ys=self.action, xs=self.network_para, grad_ys=action_grad)
            train_opt = self.optimizer.apply_gradients(zip(policy_grad, self.network_para))
            sess.run(tf.global_variables_initializer())
            sess.run(train_opt, feed_dict={self.state: s})

        # Update Target Network Parameter
        else:
            sess.run([tf.assign(t, tau * t + (1-tau) * e)
                      for t, e in zip(self.network_para, e_p)])
