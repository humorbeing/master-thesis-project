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


class RNN(torch.nn.Module):
    def __init__(self, num_actions=2, z_size=32, hidden_size=32):
        super(RNN, self).__init__()
        self.z_size=z_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.model_lstm = nn.LSTM(
            input_size=self.z_size + self.num_actions,
            hidden_size=self.hidden_size,
            # num_layers=2,
            batch_first=True
        )
        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        self.mdn_linear = nn.Linear(self.hidden_size, NOUT)
        # self.mdn_linear = nn.Sequential(
        #     nn.Linear(self.hidden_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, NOUT),
        #     # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
        #     # nn.ReLU(),
        #     # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
        #     # nn.ReLU()
        # )
        self.recon_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.z_size),
            # nn.ReLU(),
            # nn.Linear(256, NOUT),
            # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            # nn.ReLU()
        )
        self.logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        self.init_()
        self.is_cuda = False

    def init_(self):
        self.h_model_LSTM = torch.zeros(1, 1, self.hidden_size)
        self.c_model_LSTM = torch.zeros(1, 1, self.hidden_size)
        return self.h_model_LSTM

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=3)
        x = torch.logsumexp(logweight, dim=3, keepdim=True)
        logweight = logweight - x
        return logweight, mean, logstd

    def mdn_loss_f(self, z):

        # print('in mdn')
        # print(z.shape)
        # print(z[0,0,0])
        z = z.unsqueeze(3)
        # print(z[0,0,0,0])
        # print(z.shape)
        # print(self.logweight_mdn.shape)

        # y = torch.reshape(z,[-1, 1])

        lognormal = z - self.mean_mdn
        # print(lognormal.shape)
        # print('now')
        lognormal = lognormal / torch.exp(self.logstd_mdn)
        lognormal = lognormal.pow(2)
        lognormal = lognormal * (-0.5)
        lognormal = lognormal - self.logstd_mdn
        lognormal = lognormal - self.logSqrtTwoPI
        v = self.logweight_mdn +lognormal
        # print(v.shape)
        # return 1
        v = torch.logsumexp(v, dim=3, keepdim=True)
        v = (-1.0) * v
        v = torch.mean(v)
        # v = (-1.0)*v
        return v
    def prediction_loss_f(self, z):
        loss = F.mse_loss(self.z_prediction, z)
        return loss

    def forward(self, z_a):
        # x_model = z_a
        # x_model = torch.cat((self.z, action_onehot), dim=1)

        # self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
        #     x_model, (self.h_model_LSTM, self.c_model_LSTM)
        # )
        if self.training:
            self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
                z_a
            )
            # if self.training:
            vecs = self.mdn_linear(self.h_model_LSTM)
            # print('vecs')
            # print(vecs.shape)
            vecs = torch.reshape(vecs, (1, -1, self.z_size, self.num_mix * 3))
            # print(vecs.shape)
            self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)

            # reconstruction double laim
            self.z_prediction = self.recon_linear(self.h_model_LSTM)
        else:
            # print(z_a.shape)
            # print('hi')
            output, (self.h_model_LSTM, self.c_model_LSTM) = self.model_lstm(
                z_a, (self.h_model_LSTM, self.c_model_LSTM)
            )
            # print(output.shape)
            # print(self.h_model_LSTM.shape)
            # print(output)
            # print(self.h_model_LSTM)
            # input('hi')
        return self.h_model_LSTM
