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
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            # nn.Linear(input_size, 64),
            # nn.ReLU()
        )
        self.lstm = nn.LSTMCell(32, self.hidden_size)
        self.fc_actor = nn.Linear(self.hidden_size, self.output_size)
        self.fc_critic = nn.Linear(self.hidden_size, 1)

        # self.apply(weights_init)
        self.fc_actor.weight.data = normalized_columns_initializer(
            self.fc_actor.weight.data, 0.01)
        self.fc_actor.bias.data.fill_(0)
        self.fc_critic.weight.data = normalized_columns_initializer(
            self.fc_critic.weight.data, 1.0)
        self.fc_critic.bias.data.fill_(0)
        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()
    def init_(self):
        self.h_lstm = torch.zeros(1, self.hidden_size)
        self.c_lstm = torch.zeros(1, self.hidden_size)

    def forward(self, inputs):
        z = self.encoder(inputs)
        # self.h_lstm, self.c_lstm = self.lstm(
        #     z, (self.h_lstm, self.c_lstm)
        # )
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
    # state = state.dtype(np.float)
    state = state / 255.0
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
num_processes = 9
seed = 9

def train(rank, shared_model, optimizer, counter, lock):
    env = gym.make(env_name)
    env.seed(seed + rank)
    torch.manual_seed(seed + rank)
    model = Model()
    model.train()

    while True:
        state = env.reset()
        model.load_state_dict(shared_model.state_dict())
        # model.init_()
        value_s = []
        action_log_prob_s = []
        entropy_s = []
        reward_s = []

        while True:
            state = tensor_state(state)
            action = model(state)

            value = model.v
            action_log_prob = model.action_log_prob
            entropy = model.entropy

            state, reward, done, _ = env.step(action)

            value_s.append(value)
            action_log_prob_s.append(action_log_prob)
            entropy_s.append(entropy)
            reward_s.append(reward)
            with lock:
                counter.value += 1
            if done:
                break

        # out of 2nd while loop
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()


def test(rank, shared_model, counter):
    with open('log.txt', 'w'):
        pass
    env = gym.make(env_name)
    env.seed(seed + rank)
    torch.manual_seed(seed + rank)

    model = Model()
    model.eval()
    start_time = time.time()
    while True:
        episode_length = 0
        model.load_state_dict(shared_model.state_dict())
        # model.init_()
        state = env.reset()
        reward_sum = 0
        while True:
            episode_length += 1
            env.render()
            state = tensor_state(state)
            action = model(state)

            state, reward, done, _ = env.step(action)
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


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    shared_model = Model()

    shared_model = shared_model.share_memory()

    optimizer = SharedAdam(shared_model.parameters(), lr=0.01)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(
            rank, shared_model, optimizer, counter, lock
        ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()