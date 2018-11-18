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

        self.vae.train()
        self.rnn.train()
        self.init_()

        self.is_cuda = False


    def load(self):
        self.vae.load_state_dict(torch.load(vae_model_path, map_location=lambda storage, loc: storage))
        self.rnn.load_state_dict(torch.load(rnn_model_path, map_location=lambda storage, loc: storage))


    def init_(self):
        self.h = self.rnn.init_()

    def forward(self, inputs):
        z = self.vae(inputs)
        return z

    def when_train(self, inputs, one, outputs):
        if self.is_cuda:
            self.vae.is_cuda = True
            self.vae.cuda()
            self.rnn.is_cuda = True
            self.rnn.cuda()


        # with torch.no_grad():
        z = self.vae(inputs)
        self.next_kl_loss = self.vae.kl_loss
        self.next_r_loss = self.vae.r_loss
        # z = z.unsqueeze(0)
        #
        #
        # z_a = torch.cat((z, one), dim=2)
        # self.rnn(z_a)
        # # z_next = self.vae(outputs)
        # self.next_kl_loss = self.vae.kl_loss
        # self.next_r_loss = self.vae.r_loss
        #
        # # z_next = z_next.unsqueeze(0)
        # z_next = z
        #
        # self.pred_loss = self.rnn.prediction_loss_f(z_next)
        # self.mdn_loss = self.rnn.mdn_loss_f(z_next)
        #
        # z_next_hat = self.rnn.z_prediction
        #
        # z_next_hat = z_next_hat.squeeze(0)
        # self.pred_recon_loss = self.vae.reconstruction_error_f(z_next_hat, inputs)


if __name__ == "__main__":
    log = LOG('vr_loss')
    is_load = True
    is_load = False
    is_cuda = True
    vr = VAERNN()
    if is_load:
        vr.load()
    if is_cuda:
        vr.cuda()
        vr.is_cuda = True

    vr.train()
    # params = list(vr.vae.parameters()) + list(vr.rnn.parameters())
    params = list(vr.vae.parameters())
    optimizer = optim.Adam(params)
    old_filelist = os.listdir(DATA_DIR)
    block = 200
    for epoch in range(10):
        # np.random.shuffle(old_filelist)
        filelist = old_filelist[0:block]
        # np.random.shuffle(filelist)
        next_kl_loss_s = []
        next_r_loss_s = []
        pred_loss_s = []
        pred_recon_loss_s = []
        mdn_loss_s = []
        for filename in filelist:

            raw_data = np.load(os.path.join(DATA_DIR, filename))
            ob_data = raw_data['obs']
            outputs = ob_data[1:, :, :, :]
            data = ob_data[:-1, :, :, :]

            actions = raw_data['action']
            actions = actions[:-1]
            data = torch.from_numpy(data)
            outputs = torch.from_numpy(outputs)
            one = one_hot(actions)
            one = torch.from_numpy(one)
            one = one.unsqueeze(0)
            one = one.type(torch.float)
            if is_cuda:
                data = data.cuda()
                one = one.cuda()
                outputs = outputs.cuda()
            vr.when_train(data, one, outputs)


            next_kl_loss = vr.next_kl_loss
            next_r_loss = vr.next_r_loss
            # pred_loss = vr.pred_loss
            # pred_recon_loss = vr.pred_recon_loss
            # mdn_loss = vr.mdn_loss
            pred_loss = 0
            pred_recon_loss = 0
            mdn_loss = 0

            next_kl_loss_s.append(next_kl_loss.item())
            next_r_loss_s.append(next_r_loss.item())
            # pred_loss_s.append(pred_loss.item())
            # pred_recon_loss_s.append(pred_recon_loss.item())
            # mdn_loss_s.append(mdn_loss.item())
            pred_loss_s.append(0)
            pred_recon_loss_s.append(0)
            mdn_loss_s.append(0)

            loss = next_r_loss + next_r_loss + pred_recon_loss + pred_loss + mdn_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        log_string = 'Epoch: {}, Next KL: {:.6f}, Next Recon Loss: {:.6f}, Pred Loss: {:.6f}, Pred Recon Loss: {:.6f}, MDN Loss: {:.6f}'.format(
            epoch,
            np.mean(next_kl_loss_s),
            np.mean(next_r_loss_s),
            np.mean(pred_loss_s),
            np.mean(pred_recon_loss_s),
            np.mean(mdn_loss_s)
        )
        log.log(log_string)
    log.end()
