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


# Hyper-parameter
LEARNING_RATE = 0.0001
num_processes = 6

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



class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.action_map = {
            0: 2,
            1: 3
        }
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

        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.kl_tolerance = 0.5


        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        self.mdn_linear = nn.Linear(256, NOUT)
        self.train()
        # self.are_we_training = True
        self.logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

    def reset_hidden_layer(self):
        batch_this = 1
        self.h_policy_LSTM = torch.zeros(batch_this, 256)
        self.c_policy_LSTM = torch.zeros(batch_this, 256)
        self.h_model_LSTM = torch.zeros(batch_this, 256)
        self.c_model_LSTM = torch.zeros(batch_this, 256)

    def encode(self, inputs):
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        self.mu = self.en_fc_mu(x)
        self.logvar = self.en_fc_log_var(x)
        self.sigma = torch.exp(self.logvar / 2.0)
        epsilon = torch.randn(*self.sigma.size())
        z = self.mu + epsilon * self.sigma
        return z

    def kl_loss(self):
        kl_loss = 1 + self.logvar
        kl_loss -= self.mu.pow(2)
        kl_loss -= self.logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        # print(kl_loss.shape)
        kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance * self.z_size]))
        # kl_loss = torch.clamp(kl_loss, min=0.5 * 32)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def reconstruction_error(self):
        x_hat = self.decode_fc(self.z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)
        # print(self.inputs.shape)
        # print(x_hat.shape)
        r_loss = F.mse_loss(x_hat, self.inputs)
        return r_loss

    def mdn_loss(self):
        # print(self.z.shape)
        # print(self.logweight_mdn.shape)
        y = torch.reshape(self.z,[-1, 1])
        # print(flat_target_data.shape)
        lognormal = y - self.mean_mdn
        lognormal = lognormal / torch.exp(self.logstd_mdn)
        lognormal = lognormal.pow(2)
        lognormal = lognormal * (-0.5)
        lognormal = lognormal - self.logstd_mdn
        lognormal = lognormal - self.logSqrtTwoPI
        v = self.logweight_mdn +lognormal
        v = torch.logsumexp(v, dim=1, keepdim=True)
        v = torch.mean(v)
        v = (-1.0)*v
        return v

    def forward(self, inputs):

        self.z = self.encode(inputs)


        x_policy = torch.cat((self.z, self.h_model_LSTM), dim=1)
        self.h_policy_LSTM, self.c_policy_LSTM = self.policy_lstm(
            x_policy, (self.h_policy_LSTM, self.c_policy_LSTM))

        logit = self.actor_linear(self.h_policy_LSTM)
        prob = F.softmax(logit, dim=1)

        if self.training:
            sample_num = prob.multinomial(num_samples=1).data
        else:
            sample_num = prob.max(1, keepdim=True)[1].data



        action = self.action_map[sample_num.item()]

        action_onehot = torch.FloatTensor(sample_num.shape[0], self.num_outputs)
        action_onehot.zero_()
        action_onehot.scatter_(1, sample_num, 1)

        x_model = torch.cat((self.z, action_onehot), dim=1)

        self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
            x_model, (self.h_model_LSTM, self.c_model_LSTM)
        )
        if self.training:
            self.inputs = inputs
            self.v = self.critic_linear(self.h_policy_LSTM)
            log_prob = torch.log(prob)
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.log_prob_action = log_prob.gather(1, sample_num)
            vecs = self.mdn_linear(self.h_model_LSTM)
            vecs = torch.reshape(vecs, (-1, self.num_mix * 3))
            self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)


        return action

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=1)
        x = torch.logsumexp(logweight, dim=1, keepdim=True)
        logweight = logweight - x
        # print('yo')
        # print(x.shape)
        return logweight, mean, logstd

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

def sumup(x):
    y = torch.zeros(1, 1)
    for i in x:
        y += i
    return y
def train(rank, shared_model, optimizer, counter, lock):
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    model.train()
    # optimizer = optim.Adam(model.parameters())
    # state = env.reset()
    # state = tensor_state(state)
    # done = True
    while True:
        # if done:
        model.load_state_dict(shared_model.state_dict())
        model.reset_hidden_layer()
        state = env.reset()
        state = tensor_state(state)

        action = model(state)
        value_s = []
        log_prob_action_s = []
        entropy_s = []
        kl_loss_s = []
        r_loss_s = []
        mdn_loss_s = []
        reward_s = []

        while True:
            # env.render()
            value = model.v
            # print(value.shape)
            # input('hi')
            log_prob_action = model.log_prob_action
            entropy = model.entropy
            kl_loss = model.kl_loss()
            r_loss = model.reconstruction_error()
            # print(action)
            state, reward, done,_ = env.step(action)
            state = tensor_state(state)
            action = model(state)
            mdn_loss = model.mdn_loss()

            value_s.append(value)
            log_prob_action_s.append(log_prob_action)
            entropy_s.append(entropy)
            kl_loss_s.append(kl_loss)
            r_loss_s.append(r_loss)
            mdn_loss_s.append(mdn_loss)
            reward_s.append(reward)
            with lock:
                counter.value += 1
            if done:
                break
        # value = model.v
        R = torch.zeros(1, 1)
        # log_prob_action = model.log_prob_action
        # entropy = model.entropy
        # kl_loss = model.kl_loss()
        # r_loss = model.reconstruction_error()
        value_s.append(R)
        log_prob_action_s.append(log_prob_action)
        entropy_s.append(entropy)
        kl_loss_s.append(kl_loss)
        r_loss_s.append(r_loss)
        # print(len(value_s))
        # print(len(mdn_loss_s))
        # print(len(reward_s))
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        gamma = 0.99
        tau = 1.00
        entropy_coef = 0.01
        value_loss_coef = 0.5
        for i in reversed(range(len(reward_s))):
            R = gamma * R + reward_s[i]
            advantage = R - value_s[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = reward_s[i] + gamma * value_s[i + 1].data - value_s[i].data
            gae = gae * gamma * tau + delta_t

            policy_loss = policy_loss - log_prob_action_s[i] * gae - entropy_coef * entropy_s[i]

        kl_loss = sumup(kl_loss_s)
        r_loss = sumup(r_loss_s)
        mdn_loss = sumup(mdn_loss_s)
        loss = policy_loss + value_loss_coef * value_loss
        loss = loss + kl_loss + r_loss + mdn_loss
        # print(mdn_loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()

def test(rank, shared_model, counter):
    env_name = 'Pong-v0'
    env = gym.make(env_name)
    env.seed(rank)
    torch.manual_seed(rank)

    model = Model(3, 2)
    # model.train()

    model.eval()


    state = env.reset()
    state = tensor_state(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=200)
    all_good = True
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
        actions.append(action)
        if actions.count(actions[0]) == actions.maxlen:
            all_good = False

        if done:
            if all_good:
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
            actions.clear()
            all_good = True
            state = env.reset()
            time.sleep(5)

        state = tensor_state(state)


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    mp.set_start_method('spawn')

    # args = get_args()
    # env = gym.make(args.env_name)

    shared_model = Model(3, 2)

    shared_model = shared_model.share_memory()
    optimizer = SharedAdam(shared_model.parameters(), lr=LEARNING_RATE)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(num_processes, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank,
                                           shared_model,
                                           optimizer,
                                           counter,
                                           lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
