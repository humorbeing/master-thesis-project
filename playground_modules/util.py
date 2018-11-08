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


def run(Model, train, test, num_processes, LEARNING_RATE):
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    shared_model = Model(3, 2)

    shared_model = shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=LEARNING_RATE)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(Model, num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(Model, rank,
                                           shared_model,
                                           optimizer,
                                           counter,
                                           lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def test(Model, rank, shared_model, counter):
    with open('log.txt', 'w'):
        pass
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    model.eval()


    state = env.reset()
    state = model.tensor_state(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    # actions = deque(maxlen=2000)
    # all_good = True
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        env.render()
        if done:
            model.load_state_dict(shared_model.state_dict())
            model.reset_hidden_layer()

        # state = env.reset()
        # state = tensor_state(state)

        action = model(state)
        # value, prob, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        #
        # action = prob.max(1, keepdim=True)[1].data

        state, reward, done, _ = env.step(action)
        # state = tensor_state(state)
        # state, reward, done, _ = env.step(action.numpy())
        # state, reward, done, _ = env.step(action.numpy()[0, 0])

        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        # actions.append(action)
        # if actions.count(actions[0]) == actions.maxlen:
        #     all_good = False

        if done:
            # if all_good:
            string = "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length)
            print(string)
            with open('log.txt', 'a') as f:
                f.write(string + '\n')
            reward_sum = 0
            episode_length = 0
            # actions.clear()
            all_good = True
            state = env.reset()
            time.sleep(5)

        state = model.tensor_state(state)

