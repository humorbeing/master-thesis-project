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
    return state

class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.num_outputs = num_outputs
        self.conv1 = nn.Conv2d(num_inputs, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 2, stride=2)
        self.z_size = 32
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU())
        self.en_fc_log_var = nn.Linear(in_features=256 * 2 * 2, out_features=self.z_size)
        self.en_fc_mu = nn.Linear(in_features=256 * 2 * 2, out_features=self.z_size)

        self.decode_fc = nn.Linear(in_features=32, out_features=1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2),
            nn.Sigmoid())
        self.policy_lstm = nn.LSTMCell(32+256, 256)
        self.model_lstm = nn.LSTMCell(32 + 2, 256)
        # num_outputs = action_space.n
        # num_outputs = action_space
        # num_outputs = 6
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)

        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)
        batch_this = 1
        self.h_policy_LSTM = torch.zeros(batch_this, 256)
        self.c_policy_LSTM = torch.zeros(batch_this, 256)
        self.h_model_LSTM = torch.zeros(batch_this, 256)
        self.c_model_LSTM = torch.zeros(batch_this, 256)

        self.train()
        self.are_we_training = True

    def forward(self, inputs):
        # inputs, (hx, cx) = inputs
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)

        mu = self.en_fc_mu(x)
        logvar = self.en_fc_log_var(x)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn(*sigma.size())
        z = mu + epsilon * sigma
        x_hat = self.decode_fc(z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)

        x_policy = torch.cat((z, self.h_model_LSTM), dim=1)
        self.h_policy_LSTM, self.c_policy_LSTM = self.policy_lstm(
            x_policy, (self.h_policy_LSTM, self.c_policy_LSTM))
        v = self.critic_linear(self.h_policy_LSTM)
        logit = self.actor_linear(self.h_policy_LSTM)
        prob = F.softmax(logit, dim=1)
        log_prob = torch.log(prob)
        if self.are_we_training:
            # print(self.training)
            action = prob.multinomial(num_samples=1).data
        else:
            action = max(5)
        log_prob_action = log_prob.gather(1, action)
        print('aaaa')
        print(action)
        print('bbbb')
        action_onehot = torch.FloatTensor(action.shape[0], self.num_outputs)
        action_onehot.zero_()
        action_onehot.scatter_(1, action, 1)
        print(action_onehot)

        x_model = torch.cat((z, action_onehot), dim=1)
        print(x_model.shape)
        self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
            x_model, (self.h_model_LSTM, self.c_model_LSTM)
        )

        return z, x_hat, mu, logvar, v, prob, action
env_name = 'Pong-v0'
env = gym.make(env_name)
print(env.action_space)
action_space = 2
action_map = {
    0: 2,
    1: 3
}
print(action_map[0])
model = Model(3, 2)

obs = env.reset()
# obs = image_pre_process(obs)
print(obs.shape)
state = tensor_state(obs)

print(state.shape)
state = state.unsqueeze(0)
# state = state.unsqueeze(0)
# c = [2,3,42,42]
# state = torch.randn(*c)
print(state.shape)
z, x_reconstruction, mu, logvar, v, prob, action = model(state)
print(z.shape)
print(x_reconstruction.shape)
y = state - x_reconstruction
print(y.shape)
r_loss = F.mse_loss(x_reconstruction, y)
print(r_loss.shape)
print('- '*20)
print(mu.shape)
print(logvar.shape)

kl_loss = 1 + logvar
kl_loss -= mu.pow(2)
kl_loss -= logvar.exp()
kl_loss = torch.sum(kl_loss, dim=1)
kl_loss *= -0.5
print(kl_loss.shape)
kl_loss = torch.max(kl_loss, torch.FloatTensor([0.5 * 32]))
# kl_loss = torch.clamp(kl_loss, min=0.5 * 32)
kl_loss = torch.mean(kl_loss)
print(kl_loss.shape)
print(r_loss)
print(kl_loss)

print('---   '*9)
# print(xx.shape)
# print(xx)
print(v.shape)
print(prob.shape)
print(action)

# batch_size = 5
# nb_digits = 10

# y = torch.LongTensor(batch_size,1).random_() % nb_digits
# y_onehot = torch.FloatTensor(batch_size, nb_digits)
# y_onehot.zero_()
# y_onehot.scatter_(1, y, 1)
# print(y)
# print(y_onehot)
