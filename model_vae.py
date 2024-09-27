import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=16, latent_dim=512):

        super().__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # (32, 32)
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(out_channels, 2*out_channels, 3, padding=1, stride=2), # (16, 16)
            nn.ReLU(),
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2*out_channels, 4*out_channels, 3, padding=1, stride=2), # (8, 8)
            nn.ReLU(),
            nn.Conv2d(4*out_channels, 4*out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(4*out_channels*8*8, latent_dim)
        self.fc_logvar = nn.Linear(4*out_channels*8*8, latent_dim)

    def forward(self, x):
        h = self.net(x)
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        return z_mean, z_logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=16, in_channels=3):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(latent_dim, 4*out_channels*8*8)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(4*out_channels, 4*out_channels, 3, padding=1), # (8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(4*out_channels, 2*out_channels, 3, padding=1, stride=2, output_padding=1), # (16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(2*out_channels, 2*out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1, stride=2, output_padding=1), # (32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1)
        )

    def forward(self, z):
        output = self.linear(z)
        output = output.view(-1, 4*self.out_channels, 8, 8)
        output = self.conv(output)
        return output


class VAE(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, out_channels, latent_dim)
        self.decoder = Decoder(latent_dim, out_channels, in_channels)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_logvar

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = (log_qzx - log_pz).sum(-1)
        return kl

