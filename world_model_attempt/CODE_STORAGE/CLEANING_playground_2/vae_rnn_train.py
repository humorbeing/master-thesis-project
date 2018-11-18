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


class VAERNN(torch.nn.Module):
    def __init__(self):
        super(VAERNN, self).__init__()

        self.z_size = 32
        self.kl_tolerance = 0.5

        self.vae = VAE()
        self.rnn = RNN()
        self.vae.load_state_dict(torch.load(vae_model_path, map_location=lambda storage, loc: storage))
        self.rnn.load_state_dict(torch.load(rnn_model_path, map_location=lambda storage, loc: storage))
        self.vae.train()
        self.rnn.train()
        self.init_()

        self.is_cuda = False

    def init_(self):
        self.h = self.rnn.init_()

    def forward(self, inputs):
        z = self.vae(inputs)
        # z = z.unsqueeze(0)
        # z = self.rnn(z)
        print('z', z.shape)
        print('h', self.h.shape)
        return z, self.h

    def when_train(self, inputs, one, outputs):

        self.vae.is_cuda = True
        self.vae.cuda()
        self.rnn.is_cuda = True
        self.rnn.cuda()

        # print('inputs outputs')
        # print(inputs.shape)
        # print(outputs.shape)
        with torch.no_grad():
            z = self.vae(inputs)
        # print(z.shape)
        z = z.unsqueeze(0)
        # print(z.shape)

        z_a = torch.cat((z, one), dim=2)
        self.rnn(z_a)
        z_next = self.vae(outputs)
        self.next_kl_loss = self.vae.kl_loss
        self.next_r_loss = self.vae.r_loss
        # print('z_next', z_next.shape)
        # print(next_kl_loss.shape)
        # print(next_r_loss.shape)
        # print('rnn now')
        # print(self.rnn.z_prediction.shape)
        z_next = z_next.unsqueeze(0)
        # print(z_next.shape)
        # input('hi')
        self.pred_loss = self.rnn.prediction_loss_f(z_next)
        self.mdn_loss = self.rnn.mdn_loss_f(z_next)
        # print(pred_loss.shape)
        # print(mdn_loss.shape)
        z_next_hat = self.rnn.z_prediction
        # print('making v m error')
        # print(z_next_hat.shape)
        # print(outputs.shape)
        z_next_hat = z_next_hat.squeeze(0)
        self.pred_recon_loss = self.vae.reconstruction_error_f(z_next_hat, inputs)
        # print(pred_recon_loss.shape)
        '''
        w = self.rnn.logweight_mdn
        m = self.rnn.mean_mdn
        s = self.rnn.logstd_mdn
        print('w', w.shape)
        print(w[0, 0, 0])
        a = w[0, 0, 0]
        b = torch.exp(a)
        print(b)
        n = b.multinomial(num_samples=1).data
        print(n)
        weight = torch.exp(w)
        ns = weight.multinomial(num_samples=1).data
        print(ns.shape)
        c = weight[0, 0]
        d = c.multinomial(num_samples=1).data
        print(c.shape)
        print(d.shape)
        weight = weight.squeeze(0)
        print('ww', weight.shape)
        a = torch.reshape(weight, (-1, 5))
        print(a.shape)
        d = a.multinomial(num_samples=5).data
        print('d is ', d.shape)
        b = torch.reshape(d, (-1, 32, 5))
        print(b.shape)
        #c = (weight==b)
        #print(c.shape)
        #print(c[200,30,4])
        c = b[:,:,0:1]
        c = c.unsqueeze(0)
        print(c[0,250,20,0])
        print(c[0,c[0,250,20,0],20,0])
        print(c.shape)
        samples = c
        # z_a = z_a.unsqueeze(0)
        '''
        # print(z_a.shape)

    def make_prediction(self, action):
        one = one_hot(action)
        one = torch.from_numpy(one)
        one = one.unsqueeze(0)
        one = one.type(torch.float)
        z_a = torch.cat((z, one), dim=1)
        z_a = z_a.unsqueeze(0)

if __name__ == "__main__":
    with open('loss_log.txt', 'w'):
        pass
    # env = gym.make('Pong-v0')
    # state1 = env.reset()
    # state1 = tensor_state(state1)
    # print(state1.shape)
    vr = VAERNN()
    vr.train()
    vr.cuda()
    optimizer = optim.Adam(vr.parameters())
    # vr.init_()
    # z, h = vr(state1)
    # vr.when_train(state1, 2)
    old_filelist = os.listdir(DATA_DIR)
    block = 20
    # N = len(filelist)

    # N = len(filelist)

    for epoch in range(50):
        np.random.shuffle(old_filelist)
        # idx = indices[0:block]
        filelist = old_filelist[0:block]
        np.random.shuffle(filelist)
        next_kl_loss_s = []
        next_r_loss_s = []
        pred_loss_s = []
        pred_recon_loss_s = []
        mdn_loss_s = []
        for filename in filelist:

            raw_data = np.load(os.path.join(DATA_DIR, filename))
            data = raw_data['obs']
            outputs = data[1:, :, :, :]
            data = data[:-1, :, :, :]

            actions = raw_data['action']
            actions = actions[:-1]
            data = torch.from_numpy(data)
            outputs = torch.from_numpy(outputs)
            one = one_hot(actions)
            # print('one', one.shape)
            one = torch.from_numpy(one)
            one = one.unsqueeze(0)
            one = one.type(torch.float)
            # print(data.shape)
            # print(actions.shape)
            data = data.cuda()
            one = one.cuda()
            outputs = outputs.cuda()
            vr.when_train(data, one, outputs)

            # print('losses')
            next_kl_loss = vr.next_kl_loss
            next_r_loss = vr.next_r_loss
            pred_loss = vr.pred_loss
            pred_recon_loss = vr.pred_recon_loss
            mdn_loss = vr.mdn_loss
            next_kl_loss_s.append(next_kl_loss.item())
            next_r_loss_s.append(next_r_loss.item())
            pred_loss_s.append(pred_loss.item())
            pred_recon_loss_s.append(pred_recon_loss.item())
            mdn_loss_s.append(mdn_loss.item())

            loss = next_r_loss + next_r_loss + pred_recon_loss + pred_loss + mdn_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(loss)
            # print(mdn_loss.item())
        log_string = 'Epoch: {}, Next KL: {:.4f}, Next Recon Loss: {:.4f}, Pred Loss: {:.4f}, Pred Recon Loss: {:.4f}, MDN Loss: {:.4f}'.format(
            epoch,
            np.mean(next_kl_loss_s),
            np.mean(next_r_loss_s),
            np.mean(pred_loss_s),
            np.mean(pred_recon_loss_s),
            np.mean(mdn_loss_s)
        )
        print(log_string)
        with open('loss_log.txt', 'a') as f:
            f.write(log_string + '\n')
    # state2,_,_,_ = env.step(2)