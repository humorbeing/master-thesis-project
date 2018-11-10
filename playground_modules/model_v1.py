import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# v1 is original idea, where I fixed action for test and fixed mdn loss for next state

class Model(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        self.action_map = {
            0: 2,
            1: 3
        }
        self.num_outputs = num_outputs
        self.NHML = 256  # number of hidden layers in Model LSTM
        self.NHPL = 256  # number of hidden layers in Policy LSTM
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
        self.policy_lstm = nn.LSTMCell(self.z_size + self.NHML, self.NHPL)
        self.model_lstm = nn.LSTMCell(self.z_size + self.num_outputs, self.NHML)
        self.critic_linear = nn.Linear(self.NHPL, 1)
        self.actor_linear = nn.Linear(self.NHPL, self.num_outputs)


        self.kl_tolerance = 0.5


        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        self.mdn_linear = nn.Linear(self.NHML, NOUT)
        self.logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        self.train()

    def reset_hidden_layer(self):
        batch_this = 1
        self.h_policy_LSTM = torch.zeros(batch_this, self.NHPL)
        self.c_policy_LSTM = torch.zeros(batch_this, self.NHPL)
        self.h_model_LSTM = torch.zeros(batch_this, self.NHML)
        self.c_model_LSTM = torch.zeros(batch_this, self.NHML)
        self.logweight_mdn = None

    def image_pre_process(self, frame):
        frame = frame[34:34 + 160, :160]
        frame = cv2.resize(frame, (80, 80))
        frame = cv2.resize(frame, (42, 42))
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.moveaxis(frame, -1, 0)
        return frame

    def tensor_state(self, state):
        state = self.image_pre_process(state)
        state = torch.from_numpy(state)
        state = state.unsqueeze(0)
        return state

    def encode(self, inputs):
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        self.mu = self.en_fc_mu(x)
        self.logvar = self.en_fc_log_var(x)
        self.sigma = torch.exp(self.logvar / 2.0)
        epsilon = torch.randn(*self.sigma.size())
        z = self.mu + epsilon * self.sigma
        return z

    def kl_loss_f(self):
        kl_loss = 1 + self.logvar
        kl_loss -= self.mu.pow(2)
        kl_loss -= self.logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        kl_loss = torch.max(kl_loss, torch.FloatTensor([self.kl_tolerance * self.z_size]))
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def reconstruction_error_f(self, inputs):
        x_hat = self.decode_fc(self.z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)
        r_loss = F.mse_loss(x_hat, inputs)
        return r_loss

    def mdn_loss_f(self):
        y = torch.reshape(self.z,[-1, 1])
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

    def one_hot_action(self, sample_num):
        action_onehot = torch.FloatTensor(sample_num.shape[0], self.num_outputs)
        action_onehot.zero_()
        action_onehot.scatter_(1, sample_num, 1)
        return action_onehot

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=1)
        x = torch.logsumexp(logweight, dim=1, keepdim=True)
        logweight = logweight - x
        return logweight, mean, logstd

    def forward(self, inputs):

        self.z = self.encode(inputs)

        # make input for policy rnn
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

        # make input for Model RNN
        action_onehot = self.one_hot_action(sample_num)
        x_model = torch.cat((self.z, action_onehot), dim=1)

        self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
            x_model, (self.h_model_LSTM, self.c_model_LSTM)
        )

        if self.training:
            self.reconstruction_loss = self.reconstruction_error_f(inputs)
            self.kl_loss = self.kl_loss_f()

            self.v = self.critic_linear(self.h_policy_LSTM)
            log_prob = torch.log(prob)
            self.entropy = -(log_prob * prob).sum(1, keepdim=True)
            self.log_prob_action = log_prob.gather(1, sample_num)
            vecs = self.mdn_linear(self.h_model_LSTM)
            vecs = torch.reshape(vecs, (-1, self.num_mix * 3))

            # if no previous state, then mdn_loss is 0
            if self.logweight_mdn is None:
                self.mdn_loss = 0
            else:
                self.mdn_loss = self.mdn_loss_f()

            # new self.z vs old mdn. then make new mdn
            self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)


        return action

