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
    def __init__(self, input_size=4, output_size=2):
        super(Model, self).__init__()
        self.action_map ={
            0: 0,
            1: 1
        }
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = 16
        self.fc_encode1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc_encode2 = nn.Linear(self.hidden_size, self.hidden_size)

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


env_name = 'CartPole-v1'
num_processes = 9
# env = gym.make(env_nam
# e)
# env.seed(1)
# torch.manual_seed(1)
#
# model = Model(4, 2)
# model.train()
#
# state = env.reset()
#
# while True:
#     env.render()
#     state = tensor_state(state)
#     action = model(state)
#     state, r, d, i = env.step(action)


def train(shared_model, optimizer, counter, lock):
    env = gym.make(env_name)
    model = Model()
    model.train()

    while True:
        state = env.reset()
        model.load_state_dict(shared_model.state_dict())

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def test(shared_model, counter):
    with open('log.txt', 'w'):
        pass
    env = gym.make(env_name)

    model = Model()
    model.eval()
    start_time = time.time()
    while True:
        episode_length = 0
        model.load_state_dict(shared_model.state_dict())
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

    p = mp.Process(target=test, args=(shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(
            shared_model, optimizer, counter, lock
        ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()