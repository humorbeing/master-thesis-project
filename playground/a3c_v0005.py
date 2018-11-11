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


def get_args():
    parser = argparse.ArgumentParser(description='A3C')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1.00,
                        help='parameter for GAE (default: 1.00)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=4,
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--max-episode-length', type=int, default=1e6,
                        help='maximum length of an episode (default: 1000000)')
    parser.add_argument('--env-name', default='Pong-v0',
                        help='environment to train on (default: PongDeterministic-v4)')
    return parser.parse_args()
# Pong-v0
# Breakout-v4
# PongDeterministic-v4

action_map = {
    0: 2,
    1: 3
}
def image_pre_process(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


def tensor_state(state):
    state = image_pre_process(state)
    state = torch.from_numpy(state)
    return state


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


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)


        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        v = self.critic_linear(x)
        logit = self.actor_linear(x)
        prob = F.softmax(logit, dim=1)
        return v, prob, (hx, cx)



def train():
    env = gym.make('Pong-v0')
    env.seed(1)
    torch.manual_seed(1)

    model = ActorCritic(1, 2)
    model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters())
    state = env.reset()

    state = tensor_state(state)
    state = state.cuda() 
    done = True
    episode_length = 0
    reward_sum = 0
    while True:
        reward_sum = 0
        if done:
            cx = torch.zeros(1, 256).cuda()
            hx = torch.zeros(1, 256).cuda()
        else:
            cx = cx.data
            hx = hx.data

        values = []
        log_probs = []
        rewards = []
        entropies = []
        while True:
            episode_length += 1

            value, prob, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))

            log_prob = torch.log(prob)
            entropy = -(log_prob * prob).sum(1, keepdim=True)

            entropies.append(entropy)
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action_map[action.item()])
            reward_sum += reward
            if done:
                episode_length = 0
                state = env.reset()

            state = tensor_state(state)
            state = state.cuda() 
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1).cuda()


        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).cuda()
        gamma = 0.99
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + gamma * values[i + 1].data - values[i].data
            gae = gae * gamma + delta_t

            policy_loss = policy_loss - log_probs[i] * gae - 0.01 * entropies[i]
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        optimizer.step()
        print(reward_sum)
train()

