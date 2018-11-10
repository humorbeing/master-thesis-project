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

def train_v1(Model, rank, shared_model, optimizer, counter, lock):
    def sumup(x):
        y = torch.zeros(1, 1)
        for i in x:
            y += i
        return y

    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    model.train()

    while True:

        model.load_state_dict(shared_model.state_dict())
        model.reset_hidden_layer()
        state = env.reset()
        state = model.tensor_state(state)

        action = model(state)
        value_s = []
        log_prob_action_s = []
        entropy_s = []
        kl_loss_s = []
        r_loss_s = []
        mdn_loss_s = []
        reward_s = []

        while True:

            value = model.v

            log_prob_action = model.log_prob_action
            entropy = model.entropy
            kl_loss = model.kl_loss
            r_loss = model.reconstruction_loss

            state, reward, done,_ = env.step(action)
            state = model.tensor_state(state)
            action = model(state)
            mdn_loss = model.mdn_loss

            value_s.append(value)
            log_prob_action_s.append(log_prob_action)
            entropy_s.append(entropy)
            kl_loss_s.append(kl_loss)
            r_loss_s.append(r_loss)
            mdn_loss_s.append(mdn_loss)
            reward_s.append(reward)
            with lock:
                counter.value += 1
            if done:
                break

        R = torch.zeros(1, 1)

        value_s.append(R)
        log_prob_action_s.append(log_prob_action)
        entropy_s.append(entropy)
        kl_loss_s.append(kl_loss)
        r_loss_s.append(r_loss)

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

            policy_loss = policy_loss - log_prob_action_s[i] * gae - entropy_coef * entropy_s[i]

        kl_loss = sumup(kl_loss_s)
        r_loss = sumup(r_loss_s)
        mdn_loss = sumup(mdn_loss_s)
        loss = policy_loss + value_loss_coef * value_loss
        loss = loss + kl_loss + r_loss + mdn_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()


def train_v2(Model, rank, shared_model, optimizer, counter, lock):
    def sumup(x):
        y = torch.zeros(1, 1)
        for i in x:
            y += i
        return y

    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    model.train()

    while True:

        model.load_state_dict(shared_model.state_dict())
        model.reset_hidden_layer()
        state = env.reset()
        state = model.tensor_state(state)

        action = model(state)
        value_s = []
        log_prob_action_s = []
        entropy_s = []
        mdn_loss_s = []
        reward_s = []

        while True:

            value = model.v

            log_prob_action = model.log_prob_action
            entropy = model.entropy

            state, reward, done,_ = env.step(action)
            state = model.tensor_state(state)
            action = model(state)
            mdn_loss = model.mdn_loss

            value_s.append(value)
            log_prob_action_s.append(log_prob_action)
            entropy_s.append(entropy)
            mdn_loss_s.append(mdn_loss)
            reward_s.append(reward)
            with lock:
                counter.value += 1
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

            policy_loss = policy_loss - log_prob_action_s[i] * gae - entropy_coef * entropy_s[i]

        mdn_loss = sumup(mdn_loss_s)
        loss = policy_loss + value_loss_coef * value_loss
        loss = loss + mdn_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()


def train_cuda_v2(Model, rank, shared_model, optimizer, counter, lock):
    def sumup(x):
        y = torch.zeros(1, 1)
        for i in x:
            y += i
        return y

    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    model.cuda()
    model.train()

    while True:

        model.load_state_dict(shared_model.state_dict())
        model.reset_hidden_layer()
        state = env.reset()
        state = model.tensor_state(state)

        action = model(state)
        value_s = []
        log_prob_action_s = []
        entropy_s = []
        pred_loss_s = []
        reward_s = []

        while True:

            value = model.v

            log_prob_action = model.log_prob_action
            entropy = model.entropy

            state, reward, done,_ = env.step(action)
            state = model.tensor_state(state)
            action = model(state)
            pred_loss = model.pred_loss

            value_s.append(value)
            log_prob_action_s.append(log_prob_action)
            entropy_s.append(entropy)
            pred_loss_s.append(pred_loss)
            reward_s.append(reward)
            with lock:
                counter.value += 1
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

            policy_loss = policy_loss - log_prob_action_s[i] * gae - entropy_coef * entropy_s[i]

        pred_loss = sumup(pred_loss_s)
        loss = policy_loss + value_loss_coef * value_loss
        loss = loss + pred_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()