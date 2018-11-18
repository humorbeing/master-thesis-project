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


# # model_save_path = os.path.join(ROOT_PATH, 'model')
# if not os.path.exists(model_save_path):
#     os.makedirs(model_save_path)
# model_path = os.path.join(model_save_path, 'vae.model')

# DIR_NAME = 'record_preprocess'
# file_path = os.path.join(ROOT_PATH, DIR_NAME)
filelist = os.listdir(DATA_DIR)


def creat_dataset(filelist, MAX=5000):
    np.random.shuffle(filelist)
    data = None
    for filename in filelist:
        onefilepath = os.path.join(DATA_DIR, filename)
        raw_data = np.load(onefilepath)['obs']
        if data is None:
            data = raw_data
        else:
            data = np.concatenate((data, raw_data), axis=0)
        print('loading:', len(data))
        if len(data)> MAX:
            break
    return data

dataset = creat_dataset(filelist)
N = len(dataset)
# N = 105
batch_size = 1500
EPOCH = 100
is_cuda = True
lr = 0.0001
is_load = True
num_batches = int(np.floor(N / batch_size))

# print(num_batches)
vae_model = VAE()
if is_load:
    vae_model.load_state_dict(torch.load(vae_model_path))
vae_model.train()
if is_cuda:
    vae_model.cuda()
    vae_model.is_cuda = True
optimizer = optim.Adam(vae_model.parameters(), lr=lr)
for epoch in range(EPOCH):
    # np.random.shuffle(dataset)
    kl_loss_s = []
    r_loss_s = []
    for idx in range(num_batches):
        batch = dataset[idx * batch_size:(idx + 1) * batch_size]
        batch = torch.from_numpy(batch)
        if is_cuda:
            batch = batch.cuda()
        # print(batch.shape)
        z = vae_model(batch)
        kl_loss = vae_model.kl_loss
        r_loss = vae_model.r_loss
        kl_loss_s.append(kl_loss.item())
        r_loss_s.append(r_loss.item())
        # print(z.shape)
        # print(kl_loss.shape)
        # print(r_loss.shape)
        loss = kl_loss + r_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print()
    print('epoch: {}, KL loss: {:.2f}, Reconstruct loss: {}'.format(
        epoch, np.mean(kl_loss_s), np.mean(r_loss_s)))
    if (epoch + 1) % 20 == 0:
        # torch.save(vae_model.state_dict(), vae_model_path)
        pass

# torch.save(vae_model.state_dict(), vae_model_path)
    # break