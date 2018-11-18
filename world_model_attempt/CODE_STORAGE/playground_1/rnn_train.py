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
from rnn_me import RNN

is_cuda = True
is_load = True
raw_data = np.load(os.path.join(SERIES_DIR, "series.npz"))
data_z = raw_data['z']
data_action = raw_data['action']
# shape is wrong because those sequence has different length
block = 500
N = len(data_action)
indices = np.random.permutation(N)
idx = indices[0:block]
data_z = data_z[idx]
data_action = data_action[idx]
N = len(data_action)

batch_size = 50
num_batchs = int(N / batch_size)
EPOCH = 100

rnn_model = RNN()
if is_load:
    rnn_model.load_state_dict(torch.load(rnn_model_path))

rnn_model.train()
if is_cuda:
    rnn_model.cuda()
    rnn_model.is_cuda = True

optimizer = optim.Adam(rnn_model.parameters())


# one_hot(data_action[0])
for epoch in range(EPOCH):
    mdn_loss_s = []
    pred_loss_s = []
    indices = np.random.permutation(N)
    for idx in indices:
        # from_to = indices[idx * batch_size : (idx + 1) * batch_size]
        z = data_z[idx]
        a = data_action[idx]
        one = one_hot(a)
        # print(z.shape)
        # print(a.shape)
        inputs = np.concatenate((z[:-1, :], one[:-1, :]), axis=1)
        outputs = z[1:, :]
        # print(inputs.shape)
        inputs = tensor_rnn_inputs(inputs)
        outputs = tensor_rnn_inputs(outputs)
        if is_cuda:
            inputs = inputs.cuda()
            outputs = outputs.cuda()
        h = rnn_model(inputs)
        # print(h.shape)
        mdn_loss = rnn_model.mdn_loss_f(outputs)
        # pred_loss = rnn_model.prediction_loss_f(outputs)
        loss = mdn_loss# + pred_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mdn_loss_s.append(mdn_loss.item())
        # pred_loss_s.append(pred_loss.item())
        # print(l.shape)
    print('epoch: {}, mdn_loss: {}'.format(
        epoch, np.mean(mdn_loss_s)))
    if (epoch + 1) % 20 == 0:
        torch.save(rnn_model.state_dict(), rnn_model_path)

torch.save(rnn_model.state_dict(), rnn_model_path)



