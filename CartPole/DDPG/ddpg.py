# -*- coding: utf-8 -*-
import time
import os
import shutil

from LunarLander.DDPG.actor import *
from LunarLander.DDPG.critic import *
from LunarLander.DDPG.replay import *

import tensorflow as tf
import gym
import numpy as np

MAX_EPISODE_NUM = 1000
MAX_STEP_NUM = 1000

REPLAY_BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 128

GAMMA = 0.9


def run(sess, env):
    # Randomly initialize Actor and Critic Network
    # Initialize Target Network with parameters of Estimated Network
    actor, critic = Actor(sess, env), Critic(sess, env)

    # Environment
    randomness = 0

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(max_buffer_size=REPLAY_BUFFER_SIZE)

    current_episode_num = 0
    total_step_num = 0
    while current_episode_num < MAX_EPISODE_NUM:
        # Initialize a random process N for action exploration
        # Receive initial observation state s

        state = env.reset()
        episode_score = 0

        start_time = time.time()
        current_step_num = 0
        while current_step_num < MAX_STEP_NUM:

            env.render()

            # Select Action from [Actor(s) + Noise]
            action = actor.choose_action("Estimate", state[np.newaxis])[0]
            # Execute this action and observe reward and new state
            new_state, reward, terminal, _ = env.step(int(action))
            # Store transition (state, action, reward, new_action) in R
            replay_buffer.store([state, action, reward, terminal, new_state])

            # Sample a random mini-batch of N transition from R
            if replay_buffer.buffer_filled():

                mini_batch = replay_buffer.sample(batch_size=MINI_BATCH_SIZE)
                # Decay Randomness
                randomness *= 0.9995

                # Calculate Target Q Value:
                # print(mini_batch['new_state'])
                target_next_action = actor.choose_action(network_type='Target',
                                                         s=mini_batch['new_state'])
                target_next_q_value = critic.get_q_value(network_type='Target',
                                                         s=mini_batch['new_state'],
                                                         a=target_next_action)
                target_q_value = []
                for i in range(MINI_BATCH_SIZE):
                    if mini_batch['terminal'][i]:
                        target_q_value.append(mini_batch['reward'][i])
                    else:
                        target_q_value.append(mini_batch['reward'][i] + GAMMA * target_next_q_value[i])

                # target_q_value = np.reshape(mini_batch['reward'], [-1, 1]) + GAMMA * target_next_q_value

                # Update Critic Network (Estimated Network) by minimizing the Loss
                # Loss = 1/N * squared_sum {Target Q value - Estimate Q value}
                estimate_q_value = critic.get_q_value(network_type='Estimate',
                                                      s=mini_batch['state'],
                                                      a=mini_batch['action'])

                critic.update(update_type='Estimate',
                              t_q_v=target_q_value,
                              s=mini_batch['state'],
                              a=mini_batch['action'],
                              iter_num=total_step_num)

                # Update the Actor Network (Estimated Network) using the sampled gradient
                actor.update(update_type='Estimate',
                             s=mini_batch['state'],
                             e_q_v=estimate_q_value,
                             a=mini_batch['action'])

                # Update the Target Actor and Critic Networks softly
                actor.update(update_type='Target')
                # target_critic.update(estimate_critic.network_para)
                critic.update(update_type='Target')

                total_step_num += 1
                break

            if terminal:
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
    # environment = gym.make('LunarLander-v2')
    environment = gym.make('CartPole-v0')
    environment = environment.unwrapped
    environment.seed(1234)
    np.random.seed(1234)
    tf.set_random_seed(1234)

    log_dir = 'log/ddpg'

    with tf.Session() as session:
        merged = tf.summary.merge_all()
        session.run(tf.global_variables_initializer())
        run(session, environment)

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        tf.summary.FileWriter(log_dir, session.graph)
        writer = tf.summary.FileWriter(logdir=log_dir)

