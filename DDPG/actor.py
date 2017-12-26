# -*- coding: utf-8 -*-
from DDPG.agent import Agent

import tensorflow as tf
import numpy as np


class Actor(Agent):
    def __init__(self, network_type, env):
        self.agent_scope = network_type + '-Actor'
        with tf.variable_scope(self.agent_scope):
            super(Actor, self).__init__(network_type, env)
            self.action = self.actor_network()
            self.optimizer = tf.train.AdamOptimizer()
            self.network_para = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.agent_scope)

    def actor_network(self):
        scope = self.network_type
        if scope == 'Estimate':
            trainable = True
        else:
            trainable = False

        with tf.variable_scope('Network', reuse=tf.AUTO_REUSE):
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

    def update(self, *, sess, s=None, a_g=None, e_p=None, tau=0.9):
        with tf.variable_scope(self.agent_scope + '/Update'):
            # Update Estimate Network Parameter
            if self.network_type == 'Estimate':
                policy_gradient = tf.gradients(ys=self.action, xs=self.network_para, grad_ys=a_g,
                                               name='Policy_Gradient')
                train_opt = self.optimizer.apply_gradients(zip(policy_gradient, self.network_para),
                                                           name='Estimate_Actor_Update')
                sess.run(tf.global_variables_initializer())
                sess.run(train_opt, feed_dict={self.state: s})

            # Update Target Network Parameter
            else:
                sess.run([tf.assign(t, tau * t + (1-tau) * e)
                          for t, e in zip(self.network_para, e_p)])
