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


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Policy(torch.nn.Module):
    def __init__(self, num_outputs=2):
        super(Policy, self).__init__()
        self.action_map = {
            0: 2,
            1: 3
        }
        self.z_size = 32
        self.h_size = 32

        self.policy = nn.Sequential(
            nn.Linear(self.z_size + self.h_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            # nn.ReLU()
        )
        self.actor = nn.Linear(16, num_outputs)
        self.critic = nn.Linear(16, 1)

        self.apply(weights_init)
        self.actor.weight.data = normalized_columns_initializer(
            self.actor.weight.data, 0.05)
        self.actor.bias.data.fill_(0)
        self.critic.weight.data = normalized_columns_initializer(
            self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

        self.is_cuda = False



    def forward(self, inputs):
        ''' inputs s + h 32 + 32
            output a int
        '''
        x = self.policy(inputs)
        logit = self.actor(x)
        prob = F.softmax(logit, dim=1)
        if self.training:
            sample_num = prob.multinomial(num_samples=1).data
        else:
            sample_num = prob.max(1, keepdim=True)[1].data
        action = self.action_map[sample_num.item()]

        if self.training:
            self.v = self.critic(x)
            log_prob = torch.log(prob)
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.action_log_prob = log_prob.gather(1, sample_num)

        return action