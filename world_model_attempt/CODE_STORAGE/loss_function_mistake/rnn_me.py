import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(torch.nn.Module):
    def __init__(self, num_actions=2, z_size=32, hidden_size=32):
        super(RNN, self).__init__()
        self.z_size=z_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        self.model_lstm = nn.LSTM(
            input_size=self.z_size + self.num_actions,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        self.num_mix = 5
        NOUT = self.num_mix * 3 * self.z_size
        self.mdn_linear = nn.Linear(self.hidden_size, NOUT)
        self.recon_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.z_size),
        )
        self.logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
        self.init_()
        self.is_cuda = False

    def init_(self):
        self.h_model_LSTM = torch.zeros(1, 1, self.hidden_size)
        self.c_model_LSTM = torch.zeros(1, 1, self.hidden_size)
        return self.h_model_LSTM

    def get_mdn_coef(self, output):
        logweight, mean, logstd = torch.split(output, self.num_mix, dim=3)
        x = torch.logsumexp(logweight, dim=3, keepdim=True)
        logweight = logweight - x
        return logweight, mean, logstd

    def mdn_loss_f(self, z):

        z = z.unsqueeze(3)
        lognormal = z - self.mean_mdn
        lognormal = lognormal / torch.exp(self.logstd_mdn)
        lognormal = lognormal.pow(2)
        lognormal = lognormal * (-0.5)
        lognormal = lognormal - self.logstd_mdn
        lognormal = lognormal - self.logSqrtTwoPI
        v = self.logweight_mdn +lognormal
        v = torch.logsumexp(v, dim=3, keepdim=True)
        v = (-1.0) * v
        v = torch.mean(v)
        return v
    def prediction_loss_f(self, z):
        loss = F.mse_loss(self.z_prediction, z)
        return loss

    def forward(self, z_a):
        if self.training:
            self.h_model_LSTM, self.c_model_LSTM = self.model_lstm(
                z_a
            )
            vecs = self.mdn_linear(self.h_model_LSTM)
            vecs = torch.reshape(vecs, (1, -1, self.z_size, self.num_mix * 3))
            self.logweight_mdn, self.mean_mdn, self.logstd_mdn = self.get_mdn_coef(vecs)
            self.z_prediction = self.recon_linear(self.h_model_LSTM)
        else:
            output, (self.h_model_LSTM, self.c_model_LSTM) = self.model_lstm(
                z_a, (self.h_model_LSTM, self.c_model_LSTM)
            )
        return self.h_model_LSTM


# training

if __name__ == '__main__':
    from this_util import *
    log = LOG('rnn_loss')
    is_cuda = True
    is_load = True
    is_save = False
    raw_data = np.load(os.path.join(SERIES_DIR, "series2.npz"))
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

    for epoch in range(EPOCH):
        mdn_loss_s = []
        pred_loss_s = []
        indices = np.random.permutation(N)
        for idx in indices:
            z = data_z[idx]
            a = data_action[idx]
            one = one_hot(a)
            inputs = np.concatenate((z[:-1, :], one[:-1, :]), axis=1)
            outputs = z[1:, :]
            inputs = tensor_rnn_inputs(inputs)
            outputs = tensor_rnn_inputs(outputs)
            if is_cuda:
                inputs = inputs.cuda()
                outputs = outputs.cuda()
            h = rnn_model(inputs)

            mdn_loss = rnn_model.mdn_loss_f(outputs)
            pred_loss = rnn_model.prediction_loss_f(outputs)
            loss = mdn_loss + pred_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mdn_loss_s.append(mdn_loss.item())
            pred_loss_s.append(pred_loss.item())
        log_string = 'epoch: {}, mdn loss: {}, prediction loss: {}'.format(
            epoch, np.mean(mdn_loss_s), np.mean(pred_loss_s))
        log.log(log_string)
        if (epoch + 1) % 20 == 0:
            if is_save:
                print('saving')
                torch.save(rnn_model.state_dict(), rnn_model_path)

    log.end()
    if is_save:
        print('saving')
        torch.save(rnn_model.state_dict(), rnn_model_path)
