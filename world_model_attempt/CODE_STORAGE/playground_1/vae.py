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


class VAE(torch.nn.Module):
    def __init__(self, num_inputs=3):
        super(VAE, self).__init__()

        self.z_size = 32
        self.kl_tolerance = 0.5
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
        self.is_cuda = False

    def kl_loss_f(self, mu, logvar):
        kl_loss = 1 + logvar
        kl_loss -= mu.pow(2)
        kl_loss -= logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        # kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance * self.z_size]))
        kl_loss = torch.clamp(kl_loss, min=self.kl_tolerance * self.z_size)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def reconstruction_error_f(self, z, inputs):
        x_hat = self.decode_fc(z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)
        r_loss = F.mse_loss(x_hat, inputs)
        return r_loss

    def forward(self, inputs):
        # print(self.is_cuda)
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        mu = self.en_fc_mu(x)
        logvar = self.en_fc_log_var(x)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn(*sigma.size())
        if self.is_cuda:
            epsilon = epsilon.cuda()
        z = mu + epsilon * sigma
        if self.training:
            self.kl_loss = self.kl_loss_f(mu, logvar)
            self.r_loss = self.reconstruction_error_f(z, inputs)
        return z