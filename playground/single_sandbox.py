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


is_cuda = False


def image_pre_process(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    # frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


def tensor_state(state):
    state = image_pre_process(state)
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)
    if is_cuda:
        state = state.cuda()
    return state

class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.action_map = {
            0: 2,
            1: 3
        }
        self.num_outputs = num_outputs
        self.NHML = 16 # number of hidden layers in Model LSTM
        self.NHPL = 16 # number of hidden layers in Policy LSTM
        self.z_size = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU())
        self.z_fc = nn.Linear(in_features=256 * 2 * 2, out_features=self.z_size)
        # self.en_fc_mu = nn.Linear(in_features=256 * 2 * 2, out_features=self.z_size)

        # self.decode_fc = nn.Linear(in_features=32, out_features=1024)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2),
        #     nn.Sigmoid())

        self.policy_lstm = nn.LSTMCell(self.z_size+self.NHML, self.NHPL)
        self.model_lstm = nn.LSTMCell(self.z_size + self.num_outputs, self.NHML)
        # num_outputs = action_space.n
        # num_outputs = action_space
        # num_outputs = 6
        self.critic_linear = nn.Linear(self.NHPL, 1)
        self.actor_linear = nn.Linear(self.NHPL, self.num_outputs)
        self.kl_tolerance = 0.5
        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)
        # batch_this = 1
        # self.h_policy_LSTM = torch.zeros(batch_this, 256)
        # self.c_policy_LSTM = torch.zeros(batch_this, 256)
        # self.h_model_LSTM = torch.zeros(batch_this, 256)
        # self.c_model_LSTM = torch.zeros(batch_this, 256)

        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        # self.mdn_linear = nn.Linear(self.NHML, NOUT)
        self.train()
        # self.are_we_training = True
        self.logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

    def reset_hidden_layer(self):
        batch_this = 1
        self.h_policy_LSTM = torch.zeros(batch_this, self.NHPL)
        self.c_policy_LSTM = torch.zeros(batch_this, self.NHPL)
        self.h_model_LSTM = torch.zeros(batch_this, self.NHML)
        self.c_model_LSTM = torch.zeros(batch_this, self.NHML)
        self.is_new_pred_loss = True

    def encode(self, inputs):
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        # self.mu = self.en_fc_mu(x)
        # self.logvar = self.en_fc_log_var(x)
        # self.sigma = torch.exp(self.logvar / 2.0)
        # epsilon = torch.randn(*self.sigma.size())
        z = self.z_fc(x)
        return z

    def kl_loss(self):
        kl_loss = 1 + self.logvar
        kl_loss -= self.mu.pow(2)
        kl_loss -= self.logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        # print(kl_loss.shape)
        kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance * self.z_size]))
        # kl_loss = torch.clamp(kl_loss, min=0.5 * 32)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def reconstruction_error(self):
        x_hat = self.decode_fc(self.z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)
        # print(self.inputs.shape)
        # print(x_hat.shape)
        r_loss = F.mse_loss(x_hat, self.inputs)
        return r_loss

    def pred_loss_f(self):
        # print(self.h_model_LSTM.shape)
        # print(self.z.shape)
        self.pred_loss = F.mse_loss(self.h_model_LSTM, self.z)


    def mdn_loss(self):
        # print(self.z.shape)
        # print(self.logweight_mdn.shape)
        y = torch.reshape(self.z,[-1, 1])
        # print(flat_target_data.shape)
        lognormal = y - self.mean_mdn
        lognormal = lognormal / torch.exp(self.logstd_mdn)
        lognormal = lognormal.pow(2)
        lognormal = lognormal * (-0.5)
        lognormal = lognormal - self.logstd_mdn
        lognormal = lognormal - self.logSqrtTwoPI
        v = self.logweight_mdn +lognormal
        v = torch.logsumexp(v, dim=1, keepdim=True)
        v = torch.mean(v)
        v = (-1.0)*v
        return v

    def forward(self, inputs):
        # if self.training:
        #     self.inputs = inputs
        self.z = self.encode(inputs)


        x_policy = torch.cat((self.z, self.h_model_LSTM), dim=1)
        self.h_policy_LSTM, self.c_policy_LSTM = self.policy_lstm(
            x_policy, (self.h_policy_LSTM, self.c_policy_LSTM))
        self.v = self.critic_linear(self.h_policy_LSTM)
        logit = self.actor_linear(self.h_policy_LSTM)
        prob = F.softmax(logit, dim=1)
        log_prob = torch.log(prob)
        if self.training:
            # print(self.training)
            sample_num = prob.multinomial(num_samples=1).data
            # print(sample_num.shape)
            # print(sample_num)
            # a = prob.max(1, keepdim=True)[1].data
            # print(a.shape)
            # print(a)
            # input('hi')
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.log_prob_action = log_prob.gather(1, sample_num)
            if self.is_new_pred_loss:
                self.pred_loss = 0
                self.is_new_pred_loss = False
            else:
                self.pred_loss_f()
        else:
            sample_num = prob.max(1, keepdim=True)[1].data
        action = self.action_map[sample_num.item()]

        # print('aaaa')
        # print(sample_num)
        # print('bbbb')
        action_onehot = torch.FloatTensor(sample_num.shape[0], self.num_outputs)
        action_onehot.zero_()
        action_onehot.scatter_(1, sample_num, 1)
        # print(action_onehot)

        x_model = torch.cat((self.z, action_onehot), dim=1)
        # print(x_model.shape)
        self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
            x_model, (self.h_model_LSTM, self.c_model_LSTM)
        )
        if self.training:
            pass
        # vecs = self.mdn_linear(self.h_model_LSTM)
        # vecs = torch.reshape(vecs, (-1, self.num_mix * 3))
        # # print(vecs.shape)
        # self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)

        # print(a.shape)
        # print(b.shape)
        # print(c.shape)
        # print(vecs.shape)
        # print(action)
        # print(action.item())
        # return z, x_hat, mu, logvar, v, prob, action
        return action

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=1)
        x = torch.logsumexp(logweight, dim=1, keepdim=True)
        logweight = logweight - x
        # print('yo')
        # print(x.shape)
        return logweight, mean, logstd

def sumup(x):
    y = torch.zeros(1, 1)
    for i in x:
        y += i
    return y

env_name = 'Pong-v0'
env = gym.make(env_name)

model = Model(3, 2)

def train():

    # model.eval()
    # print(model.training)
    # input('hi')
    optimizer = optim.Adam(model.parameters())
    # state = env.reset()
    # state = tensor_state(state)
    # done = True
    while True:
        # if done:
        model.reset_hidden_layer()
        state = env.reset()
        state = tensor_state(state)

        action = model(state)
        value_s = []
        log_prob_action_s = []
        entropy_s = []
        kl_loss_s = []
        r_loss_s = []
        mdn_loss_s = []
        reward_s = []

        while True:
            # env.render()
            value = model.v
            # print(value.shape)
            # input('hi')
            log_prob_action = model.log_prob_action
            entropy = model.entropy
            # kl_loss = model.kl_loss()
            # r_loss = model.reconstruction_error()
            # print(action)
            state, reward, done,_ = env.step(action)
            state = tensor_state(state)
            action = model(state)
            model.pred_loss()
            break
            # mdn_loss = model.mdn_loss()

            value_s.append(value)
            log_prob_action_s.append(log_prob_action)
            entropy_s.append(entropy)
            kl_loss_s.append(kl_loss)
            r_loss_s.append(r_loss)
            mdn_loss_s.append(mdn_loss)
            reward_s.append(reward)

            if done:
                break
        break
        # value = model.v
        R = torch.zeros(1, 1)
        # log_prob_action = model.log_prob_action
        # entropy = model.entropy
        # kl_loss = model.kl_loss()
        # r_loss = model.reconstruction_error()
        value_s.append(R)
        log_prob_action_s.append(log_prob_action)
        entropy_s.append(entropy)
        kl_loss_s.append(kl_loss)
        r_loss_s.append(r_loss)
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

            policy_loss = policy_loss - log_prob_action_s[i] * gae - entropy_coef * entropy_s[i]

        kl_loss = sumup(kl_loss_s)
        r_loss = sumup(r_loss_s)
        mdn_loss = sumup(mdn_loss_s)
        loss = policy_loss + value_loss_coef * value_loss
        loss = loss + kl_loss + r_loss + mdn_loss
        print(mdn_loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()
# train()

while True:
    # if done:
    model.reset_hidden_layer()
    state = env.reset()
    state = tensor_state(state)

    action = model(state)
    # value_s = []
    # log_prob_action_s = []
    # entropy_s = []
    # kl_loss_s = []
    # r_loss_s = []
    # mdn_loss_s = []
    # reward_s = []

    while True:
        # env.render()
        value = model.v
        # print(value.shape)
        # input('hi')
        log_prob_action = model.log_prob_action
        entropy = model.entropy
        # kl_loss = model.kl_loss()
        # r_loss = model.reconstruction_error()
        # print(action)
        state, reward, done,_ = env.step(action)
        state = tensor_state(state)
        action = model(state)
        print(model.pred_loss)
        # break
        # # mdn_loss = model.mdn_loss()
        #
        # value_s.append(value)
        # log_prob_action_s.append(log_prob_action)
        # entropy_s.append(entropy)
        # kl_loss_s.append(kl_loss)
        # r_loss_s.append(r_loss)
        # mdn_loss_s.append(mdn_loss)
        # reward_s.append(reward)

        if done:
            break
    # break