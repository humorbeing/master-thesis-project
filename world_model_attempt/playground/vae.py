import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(torch.nn.Module):
    def __init__(self, num_inputs=3):
        super(VAE, self).__init__()

        self.z_size = 32
        self.kl_tolerance = 0.5
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
        self.is_cuda = False

    def kl_loss_f(self, mu, logvar):
        kl_loss = 1 + logvar
        kl_loss -= mu.pow(2)
        kl_loss -= logvar.exp()
        kl_loss = torch.sum(kl_loss, dim=1)
        kl_loss *= -0.5
        kl_loss = torch.clamp(kl_loss, min=self.kl_tolerance * self.z_size)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    def reconstruction_error_f(self, z, inputs):
        x_hat = self.decode_fc(z)
        x_hat = x_hat[:, :, None, None]
        x_hat = self.decoder(x_hat)
        r_loss = F.mse_loss(x_hat, inputs)
        return r_loss

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = x.view(x.size(0), -1)
        mu = self.en_fc_mu(x)
        logvar = self.en_fc_log_var(x)
        sigma = torch.exp(logvar / 2.0)
        epsilon = torch.randn(*sigma.size())
        if self.is_cuda:
            epsilon = epsilon.cuda()
        z = mu + epsilon * sigma
        if self.training:
            self.kl_loss = self.kl_loss_f(mu, logvar)
            self.r_loss = self.reconstruction_error_f(z, inputs)
        return z


# training vae
if __name__ == '__main__':
    from this_util import *
    filelist = os.listdir(DATA_DIR)
    log = LOG('vae_loss')

    def creat_dataset(filelist, MAX=500000):
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
    batch_size = 1500
    EPOCH = 1
    is_cuda = True
    lr = 0.0001
    is_load = False
    # is_load = True
    is_save = False
    num_batches = int(np.floor(N / batch_size))

    vae_model = VAE()
    if is_load:
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=lambda storage, loc: storage))
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
            z = vae_model(batch)
            kl_loss = vae_model.kl_loss
            r_loss = vae_model.r_loss
            kl_loss_s.append(kl_loss.item())
            r_loss_s.append(r_loss.item())
            loss = kl_loss + r_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log_string = 'epoch: {}, KL loss: {:.2f}, Reconstruct loss: {}'.format(
            epoch, np.mean(kl_loss_s), np.mean(r_loss_s))
        log.log(log_string)
        if (epoch + 1) % 20 == 0:
            if is_save:
                torch.save(vae_model.state_dict(), vae_model_path)
    log.end()
    if is_save:
        torch.save(vae_model.state_dict(), vae_model_path)
