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


# ROOT_PATH = '/media/ray/SSD/workspace/python/dataset/world_model_attempt'
vae_model_path = '/media/ray/SSD/workspace/python/dataset/world_model_attempt/model/vae.model'
rnn_model_path = '/media/ray/SSD/workspace/python/dataset/world_model_attempt/model/rnn.model'
# DATA_DIR = '/media/ray/SSD/workspace/python/dataset/world_model_attempt/record_preprocess'
DATA_DIR = '/media/ray/SSD/workspace/python/dataset/world_model_attempt/record_small'
SERIES_DIR = '/media/ray/SSD/workspace/python/dataset/world_model_attempt/series'


def ss(s=''):
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    print('        >>>>>>>>>>>>>>>>>>>>        '+s+'        <<<<<<<<<<<<<<<<<<<<        ')
    print()
    print('   ---' * 15)
    print('   ---' * 15)
    print()
    import sys
    sys.exit()



class LOG():
    def __init__(self, name):
        from datetime import datetime
        surfix = datetime.now().strftime('-%Y%m%d-%H-%M-%S')
        self.log_file = 'logs/' + name + surfix + '.txt'
        if not os.path.exists('logs'):
            os.makedirs('logs')
        with open(self.log_file, 'w'):
            print('opening log file:', self.log_file)

    def log(self, log_string):
        print(log_string)
        with open(self.log_file, 'a') as f:
            f.write(log_string + '\n')

    def end(self):
        print('log is saved in: {}'.format(self.log_file))



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
    return state

def tensor_rnn_inputs(inputs):
    state = torch.from_numpy(inputs)
    state = state.unsqueeze(0)
    state = state.type(torch.float)
    return state

def one_hot(a):
    a = a - 2
    one = np.eye(2)[a]
    return one

class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad