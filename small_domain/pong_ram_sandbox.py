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

from util import *

class Model(nn.Module):
    def __init__(self, input_size=128, output_size=2):
        super(Model, self).__init__()
        self.action_map ={
            0: 2,
            1: 3
        }
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 16
        self.fc_encode1 = nn.Linear(self.input_size, 64)
        self.fc_encode2 = nn.Linear(64, self.hidden_size)

        self.fc_actor = nn.Linear(self.hidden_size, self.output_size)
        self.fc_critic = nn.Linear(self.hidden_size, 1)

        self.train()

    def forward(self, inputs):
        z = F.relu(self.fc_encode1(inputs))
        z = F.relu(self.fc_encode2(z))

        logit = self.fc_actor(z)

        prob = F.softmax(logit, dim=1)

        if self.training:
            sample_num = prob.multinomial(num_samples=1).data
        else:
            sample_num = prob.max(1, keepdim=True)[1].data

        action = self.action_map[sample_num.item()]

        if self.training:
            self.v = self.fc_critic(z)
            log_prob = torch.log(prob)
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.action_log_prob = log_prob.gather(1, sample_num)

        return action


def pre_process(state):
    return state

def tensor_state(state):
    state = pre_process(state)
    # from numpy array to tensor and add a batch column
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)
    # from numpy array to tensor and add a batch column
    # state = state[None, :]
    # state = torch.Tensor(state) # not a good idea
    # state = state.type(torch.float)
    # state = state.type(torch.double)
    state = state.type(torch.FloatTensor)
    return state


env_name = 'Pong-ram-v0'

env = gym.make(env_name)

state = env.reset()
while True:
    state = state / 255.0
    print(state)
    print(max(state), min(state))
    a,b,c,d = env.step(env.action_space.sample())
    input('hi')
    if c:
        state = env.reset()

