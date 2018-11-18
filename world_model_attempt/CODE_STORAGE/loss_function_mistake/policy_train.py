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

env = gym.make('Pong-v0')

vae_model = VAE()
vae_model.load_state_dict(torch.load(vae_model_path))
vae_model.eval()

rnn_model = RNN()
rnn_model.load_state_dict(torch.load(rnn_model_path))
rnn_model.eval()

policy = Policy()
policy.train()
optimizer = optim.Adam(policy.parameters())
while True:
    state = env.reset()
    h = rnn_model.init_()
    value_s = []
    action_log_prob_s = []
    entropy_s = []
    reward_s = []
    reward_sum = 0
    while True:
        # env.render()
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

        value = policy.v
        action_log_prob = policy.action_log_prob
        entropy = policy.entropy
        state, reward, done, _ = env.step(a)
        reward_sum += reward
        value_s.append(value)
        action_log_prob_s.append(action_log_prob)
        entropy_s.append(entropy)
        reward_s.append(reward)
        if done:
            break

    R = torch.zeros(1, 1)

    value_s.append(R)


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
    print(reward_sum)