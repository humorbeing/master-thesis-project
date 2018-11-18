import argparse
import os
import gym
import numpy as np
import math
import cv2
import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim

from this_util import *
from vae import VAE
from rnn_me import RNN
from policy import Policy

vae_model = VAE()
vae_model.load_state_dict(torch.load(vae_model_path, map_location=lambda storage, loc: storage)) #, map_location=lambda storage, loc: storage)

vae_model.eval()

rnn_model = RNN()
rnn_model.load_state_dict(torch.load(rnn_model_path, map_location=lambda storage, loc: storage))
rnn_model.eval()


def wow(state, h, policy, vae_model, rnn_model):
    state = tensor_state(state)
    z = vae_model(state)
    h = h.squeeze(0)
    z_h = torch.cat((z, h), dim=1)
    a = policy(z_h)

    one = one_hot(a)
    one = torch.from_numpy(one)
    one = one.unsqueeze(0)
    one = one.type(torch.float)
    z_a = torch.cat((z, one), dim=1)
    z_a = z_a.unsqueeze(0)
    h = rnn_model(z_a)
    return a, h

def train(rank, shared_model, optimizer, counter, lock):
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    policy = Policy()
    policy.train()
    policy.cpu()


    while True:

        policy.load_state_dict(shared_model.state_dict())
        h = rnn_model.init_()
        state = env.reset()

        value_s = []
        action_log_prob_s = []
        entropy_s = []
        reward_s = []

        while True:
            a, h = wow(state, h, policy, vae_model, rnn_model)

            value = policy.v
            action_log_prob = policy.action_log_prob
            entropy = policy.entropy
            state, reward, done, _ = env.step(a)
            # reward_sum += reward
            value_s.append(value)
            action_log_prob_s.append(action_log_prob)
            entropy_s.append(entropy)
            reward_s.append(reward)
            with lock:
                counter.value += 1
            if done:
                break
        # value = model.v
        R = torch.zeros(1, 1)
        # log_prob_action = model.log_prob_action
        # entropy = model.entropy
        # kl_loss = model.kl_loss()
        # r_loss = model.reconstruction_error()
        value_s.append(R)

        # print(len(value_s))
        # print(len(mdn_loss_s))
        # print(len(reward_s))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        gamma = 0.99
        tau = 1.00
        entropy_coef = 0.01
        value_loss_coef = 0.5
        for i in reversed(range(len(reward_s))):
            R = gamma * R + reward_s[i]
            advantage = R - value_s[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = reward_s[i] + gamma * value_s[i + 1].data - value_s[i].data
            gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - action_log_prob_s[i] * gae - entropy_coef * entropy_s[i]

        loss = policy_loss + value_loss_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 50)
        optimizer.step()

def test(rank, shared_model, counter):
    with open('log.txt', 'w'):
        pass
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    policy = Policy()
    policy.eval()
    policy.cpu()

    start_time = time.time()
    while True:

        policy.load_state_dict(shared_model.state_dict())
        h = rnn_model.init_()
        state = env.reset()
        reward_sum = 0
        episode_length = 0

        while True:
            env.render()
            a, h = wow(state, h, policy, vae_model, rnn_model)


            state, reward, done, _ = env.step(a)

            reward_sum += reward

            if done:
                string = "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    counter.value, counter.value / (time.time() - start_time),
                    reward_sum, episode_length)
                print(string)
                with open('log.txt', 'a') as f:
                    f.write(string + '\n')

                time.sleep(5)
                break



num_processes = 2
LEARNING_RATE = 0.0001

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    # args = get_args()
    # env = gym.make(args.env_name)

    shared_model = Policy()

    shared_model = shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=LEARNING_RATE)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank,
                                           shared_model,
                                           optimizer,
                                           counter,
                                           lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
