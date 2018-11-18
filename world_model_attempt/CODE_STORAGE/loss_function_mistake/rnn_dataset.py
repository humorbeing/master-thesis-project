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

is_cuda = True
vae_model = VAE()
vae_model.load_state_dict(torch.load(vae_model_path))
vae_model.eval()
if is_cuda:
    vae_model.cuda()
    vae_model.is_cuda = True
filelist = os.listdir(DATA_DIR)
filelist.sort()
N = len(filelist)

z_list = []
action_list = []
for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data = raw_data['obs']
    data = torch.from_numpy(data)
    if is_cuda:
        data = data.cuda()
    t = vae_model(data)
    if is_cuda:
        t = t.cpu()
    z = t.data.numpy()
    z_list.append(z)
    action_list.append(raw_data['action'])
    # print(action_list)
    # print(data.shape)
    if ((i + 1) % 10 == 0):
        print("loading file", (i + 1))

z_list = np.array(z_list)
action_list = np.array(action_list)
np.savez_compressed(
    os.path.join(SERIES_DIR, "series2.npz"),
    action=action_list,
    z=z_list)


