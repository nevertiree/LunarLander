# -*- coding: utf-8 -*-
import time
import os
import shutil

from DDPG.actor import *
from DDPG.critic import *
from DDPG.replay import *

import tensorflow as tf
import gym
import numpy as np

MAX_EPISODE_NUM = 2
MAX_STEP_NUM = 100

GAMMA = 0.9


def run(sess, env):
    # Randomly initialize Actor and Critic Network
    estimate_actor, estimate_critic = Actor('Estimate', env), Critic('Estimate', env)
    # Initialize Target Network with parameters of Estimated Network
    target_actor, target_critic = Actor('Target', env), Critic('Target', env)

    # Environment
    randomness = 0

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer()

    current_episode_num = 0
    while current_episode_num < MAX_EPISODE_NUM:
        # Initialize a random process N for action exploration
        # Receive initial observation state s

        state = env.reset()
        episode_score = 0

        start_time = time.time()
        current_step_num = 0
        while current_step_num < MAX_STEP_NUM:

            # env.render()

            # Select Action from [Actor(s) + Noise]
            action = estimate_actor.choose_action(sess, state[np.newaxis])[0]
            # Execute this action and observe reward and new state
            new_state, reward, done, _ = env.step(int(action))
            # Store transition (state, action, reward, new_action) in R
            replay_buffer.store([state, action, reward, new_state])

            # Sample a random mini-batch of N transition from R
            if replay_buffer.buffer_filled() or done:

                mini_batch = replay_buffer.sample()
                # Decay Randomness
                randomness *= 0.9995

                # Calculate Target Q Value:
                target_next_action = target_actor.choose_action(sess=sess, s=mini_batch['new_state'])
                target_next_q_value = target_critic.get_q_value(sess=sess, s=mini_batch['new_state'],
                                                                a=target_next_action)
                target_q_value = np.reshape(mini_batch['reward'], [-1, 1]) + GAMMA * target_next_q_value

                # Update Critic Network (Estimated Network) by minimizing the Loss
                # Loss = 1/N * squared_sum {Target Q value - Estimate Q value}
                estimate_critic.update(sess=sess, t_q_v=target_q_value,
                                       s=mini_batch['state'], a=mini_batch['action'])

                # Update the Actor Network (Estimated Network) using the sampled gradient
                estimate_actor.update(sess=sess,
                                      s=mini_batch['state'],
                                      a_g=tf.gradients(estimate_critic.q_value,
                                                       mini_batch['action'],
                                                       name='Action_Gradient'))

                # Update the Target Actor and Critic Networks softly
                target_actor.update(sess=sess, e_p=estimate_actor.network_para)
                # target_critic.update(estimate_critic.network_para)
                target_critic.update(sess=sess, e_p=estimate_critic.network_para)

                break

            # End if
            episode_score += reward
            state = new_state
            current_step_num += 1

        # End for
        end_time = time.time()
        # tf.summary.scalar('Episode_Score', episode_score)
        print("Current episode num is %d, cost time is %f" % (current_episode_num, end_time-start_time))
        current_episode_num += 1

    # End def


if __name__ == '__main__':
    environment = gym.make('LunarLander-v2')
    environment = environment.unwrapped
    environment.seed(1)

    log_dir = 'log/'

    with tf.Session() as session:
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        run(session, environment)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tf.summary.FileWriter(log_dir, session.graph)
        writer = tf.summary.FileWriter(logdir='log/')


