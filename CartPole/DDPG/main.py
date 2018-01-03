# -*- coding: utf-8 -*-

from CartPole.DDPG.ddpg import *

import tensorflow as tf
import numpy as np
import gym


def main(_):
    environment = gym.make('CartPole-v0')
    environment = environment.unwrapped
    environment.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    log_dir = 'log/ddpg'

    with tf.Session() as session:
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        run(session, environment, actor_learning_rate=0.001, critic_learning_rate=0.001)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tf.summary.FileWriter(log_dir, session.graph)
        writer = tf.summary.FileWriter(logdir=log_dir)


if __name__ == '__main__':
    tf.app.run()
