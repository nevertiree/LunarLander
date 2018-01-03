# -*- coding: utf-8 -*-
import time

from Util.analysis import *
from CartPole.DDPG.actor import *
from CartPole.DDPG.critic import *
from CartPole.DDPG.replay import *

import numpy as np

MAX_EPISODE_NUM = 2000
MAX_STEP_NUM = 1000

REPLAY_BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 128

GAMMA = 0.9


def run(sess, env, actor_learning_rate=0.0001, critic_learning_rate=0.0001):
    # Randomly initialize Actor and Critic Network
    actor = Actor(sess, env, learning_rate=actor_learning_rate)
    critic = Critic(sess, env, learning_rate=critic_learning_rate)

    # Initialize Replay Buffer
    replay_buffer = ReplayBuffer(max_buffer_size=REPLAY_BUFFER_SIZE)

    fail_episode_num = 0
    total_step_num = 0
    # Data Analysis
    episode_score_list = []
    num_0 = 0
    num_1 = 0

    # Iterate N episodes
    current_episode_num = 0
    while current_episode_num < MAX_EPISODE_NUM:
        episode_score = 0
        state = env.reset()

        start_time = time.time()

        # Each episode only has M steps
        current_step_num = 0
        while current_step_num < MAX_STEP_NUM:

            env.render()

            # Select Action from [Actor(s) + Noise]
            action = actor.choose_action("Estimate", state[np.newaxis])
            if action[0] == 0:
                num_0 += 1
            else:
                num_1 += 1

            # Execute this action and observe reward and new state
            new_state, reward, terminal, _ = env.step(action[0])
            # Store transition (state, action, reward, terminal, new_action) in R
            replay_buffer.store([state, action, reward, terminal, new_state])

            # Sample a random mini-batch of N transition from Replay Buffer
            if replay_buffer.size() > MINI_BATCH_SIZE:

                mini_batch = replay_buffer.sample(batch_size=MINI_BATCH_SIZE)

                # Calculate Target Q Value:
                target_next_action = actor.choose_action(network_type='Target',
                                                         state=mini_batch['new_state'])
                target_next_q_value = critic.get_q_value(network_type='Target',
                                                         state=mini_batch['new_state'],
                                                         action=target_next_action)
                target_q_value = []
                for i in range(MINI_BATCH_SIZE):
                    if mini_batch['terminal'][i]:
                        target_q_value.append(mini_batch['reward'][i])
                    else:
                        target_q_value.append(mini_batch['reward'][i] + GAMMA * target_next_q_value[i])

                # Loss = 1/N * squared_sum {Target Q value - Estimate Q value}
                target_q_value = np.reshape(target_q_value, (-1, 1))
                estimate_q_value = critic.get_q_value(network_type='Estimate',
                                                      state=mini_batch['state'],
                                                      action=mini_batch['action'])

                # Update Critic Network (Estimated Network) by minimizing Loss Function
                critic.update(update_type='Estimate',
                              target_q_value=target_q_value,
                              state=mini_batch['state'],
                              action=mini_batch['action'],
                              iter_num=total_step_num)

                # Update the Actor Network (Estimated Network) using the sampled gradient
                actor.update(update_type='Estimate',
                             estimate_q_value=estimate_q_value,
                             state=mini_batch['state'],
                             action=mini_batch['action'])

                # Update the Target Actor and Critic Networks softly
                actor.update(update_type='Target')
                # target_critic.update(estimate_critic.network_para)
                critic.update(update_type='Target', iter_num=total_step_num)

                total_step_num += 1
                break

            if terminal:
                fail_episode_num += 1
                break

            # End if
            episode_score += reward
            state = new_state
            current_step_num += 1

        # End for
        end_time = time.time()
        # tf.summary.scalar('Episode_Score', episode_score)
        episode_score_list.append(episode_score)
        print("Current episode num is %d, cost time is %f" % (current_episode_num+1, end_time-start_time))
        current_episode_num += 1

    print(fail_episode_num)
    broke_line(episode_score_list)
    bar_chart([num_0, num_1])
    # End def
