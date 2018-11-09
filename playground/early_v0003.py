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
        # num_outputs = action_space.n
        # num_outputs = action_space
        # num_outputs = 6
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.kl_tolerance = 0.5
        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)
        # batch_this = 1
        # self.h_policy_LSTM = torch.zeros(batch_this, 256)
        # self.c_policy_LSTM = torch.zeros(batch_this, 256)
        # self.h_model_LSTM = torch.zeros(batch_this, 256)
        # self.c_model_LSTM = torch.zeros(batch_this, 256)

        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        self.mdn_linear = nn.Linear(256, NOUT)
        self.train()
        self.are_we_training = True
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
        if self.are_we_training:
            self.inputs = inputs
        self.z = self.encode(inputs)


        x_policy = torch.cat((self.z, self.h_model_LSTM), dim=1)
        self.h_policy_LSTM, self.c_policy_LSTM = self.policy_lstm(
            x_policy, (self.h_policy_LSTM, self.c_policy_LSTM))
        self.v = self.critic_linear(self.h_policy_LSTM)
        logit = self.actor_linear(self.h_policy_LSTM)
        prob = F.softmax(logit, dim=1)
        log_prob = torch.log(prob)
        if self.are_we_training:
            # print(self.training)
            sample_num = prob.multinomial(num_samples=1).data
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.log_prob_action = log_prob.gather(1, sample_num)
        else:
            action = max(5)
        action = self.action_map[sample_num.item()]

        # print('aaaa')
        # print(sample_num)
        # print('bbbb')
        action_onehot = torch.FloatTensor(sample_num.shape[0], self.num_outputs)
        action_onehot.zero_()
        action_onehot.scatter_(1, sample_num, 1)
        # print(action_onehot)

        x_model = torch.cat((self.z, action_onehot), dim=1)
        # print(x_model.shape)
        self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
            x_model, (self.h_model_LSTM, self.c_model_LSTM)
        )
        vecs = self.mdn_linear(self.h_model_LSTM)
        vecs = torch.reshape(vecs, (-1, self.num_mix * 3))
        # print(vecs.shape)
        self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)

        # print(a.shape)
        # print(b.shape)
        # print(c.shape)
        # print(vecs.shape)
        # print(action)
        # print(action.item())
        # return z, x_hat, mu, logvar, v, prob, action
        return action

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=1)
        x = torch.logsumexp(logweight, dim=1, keepdim=True)
        logweight = logweight - x
        # print('yo')
        # print(x.shape)
        return logweight, mean, logstd

def sumup(x):
    y = torch.zeros(1, 1)
    for i in x:
        y += i
    return y
def train():
    env_name = 'Pong-v0'
    env = gym.make(env_name)

    model = Model(3, 2)
    optimizer = optim.Adam(model.parameters())
    # state = env.reset()
    # state = tensor_state(state)
    # done = True
    while True:
        # if done:
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
        print(mdn_loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()
train()

'''/media/ray/SSD/workspace/python/ENVIRONMENT/python3.5/bin/python3.5 /media/ray/SSD/workspace/python/Observe_RL/playground/sendbox.py
2209.255859375
2002.573486328125
2062.91357421875
1823.9141845703125
1427.6298828125
1680.0694580078125
1731.3211669921875
1427.0338134765625
1443.6927490234375
1196.9090576171875
1470.2103271484375
1126.939208984375
1150.7401123046875
1185.693603515625
944.14990234375
1423.58251953125
1020.5972900390625
1082.368408203125
968.5101318359375
809.5315551757812
613.085205078125
671.6913452148438
484.2575988769531
745.1591186523438
1409.1173095703125
1149.81201171875
1161.833251953125
763.8815307617188
660.3465576171875
928.1162719726562
892.1175537109375
912.2120971679688
759.3568725585938
966.0023193359375
724.3278198242188
732.0003662109375
836.3831787109375
681.9600830078125
682.9187622070312
600.8739624023438
532.2359008789062
817.9711303710938
785.2003173828125
798.12744140625
773.4105224609375
571.22314453125
691.5338745117188
632.5182495117188
727.46240234375
1096.1912841796875
680.6519165039062
988.3600463867188
730.9271850585938
511.0066833496094
635.24853515625
596.6061401367188
649.3634033203125
848.4030151367188
855.62890625
630.9821166992188
624.47314453125
599.7431030273438
576.9091796875
615.4712524414062
556.63818359375
719.6198120117188
806.6124877929688
787.3780517578125
570.8167114257812
488.2840576171875
613.572265625
817.85009765625
717.064697265625
460.44744873046875
506.11688232421875
465.47723388671875
640.2860717773438
819.6749877929688
841.613525390625
1001.6048583984375
763.1406860351562
622.4126586914062
498.4680480957031
411.6566162109375
401.8402099609375
395.33868408203125
457.0047302246094
558.2584838867188
690.777099609375
685.5515747070312
651.8360595703125
527.1126708984375
665.2297973632812
589.3564453125
545.4338989257812
540.00390625
603.9466552734375
557.0999755859375
518.5076904296875
567.105712890625
488.0226745605469
752.7748413085938
738.2407836914062
763.8513793945312
484.48919677734375
683.489990234375
647.6279296875
659.6362915039062
619.4866333007812
504.9701232910156
548.3370971679688
471.061279296875
843.901611328125
548.7545166015625
500.6016540527344
550.6656494140625
493.19830322265625
536.976806640625
455.4026184082031
473.07452392578125
478.72149658203125
464.2269592285156

Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
'''