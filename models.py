import torch
import torch.nn as nn

torch.set_default_dtype(torch.float)


def cnn_output_size(input_dim=28):
    dim = input_dim
    for i in range(3):
        dim = (dim - 3 + 2 * 1) / 2 + 1
    return int(dim)


class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 image_dim: int = 28,
                 beta: int = 1):
        super(BetaVAE, self).__init__()

        self.beta = beta
        self.latent_dim = latent_dim

        cnn_channels = [32, 32, 64, 3]
        self.channels_into_decoder = cnn_channels[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.LeakyReLU(),

            nn.Conv2d(cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.LeakyReLU(),

            nn.Conv2d(cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(cnn_channels[2], out_channels=cnn_channels[1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(cnn_channels[1], out_channels=cnn_channels[0], kernel_size=3, stride=2,
                               padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(cnn_channels[0], out_channels=cnn_channels[3], kernel_size=3, stride=2,
                               padding=1),
            nn.BatchNorm2d(cnn_channels[3]),
            nn.LeakyReLU(),

            nn.Conv2d(cnn_channels[3], out_channels=in_channels,
                      kernel_size=4, padding=1),
            nn.Tanh()
        )

        # We need two Linear layers to convert encoder -> mu, sigma
        # But first we need to calculate how big the output from our network is.
        self.cnn_output_size = cnn_output_size(image_dim)
        encoder_output_size = cnn_channels[2] * self.cnn_output_size**2

        self.linear_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.linear_logsigma = nn.Linear(encoder_output_size, self.latent_dim)

        self.upsample = nn.Linear(self.latent_dim, encoder_output_size)

    def encode(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)

        mu = self.linear_mu(z)
        logsigma = self.linear_logsigma(z)

        return mu, logsigma

    def decode(self, x):
        z = self.upsample(x)
        z = z.view(-1, self.channels_into_decoder, self.cnn_output_size, self.cnn_output_size)
        z = self.decoder(z)

        return z

    def forward(self, x):
        mu, logsigma = self.encode(x)
        # This is done to make sure we get a positive semi-definite cov-matrix.
        sigma = torch.exp(logsigma)

        # The reparameterization-trick
        z_tmp = torch.normal(torch.zeros_like(mu), 1)
        z = mu + sigma * z_tmp

        output = self.decode(z)

        return output, (mu, sigma)

    def compute_loss(self, x, output, mu, logsigma):
        # First we compare how well we have recreated the image
        mse_loss = nn.functional.mse_loss(output, x)

        # Then the KL_divergence
        kl_div = 0.5 * (torch.exp(logsigma)**2 + mu**2 - 2*logsigma - 1).sum(axis=1)

        loss = mse_loss + self.beta*kl_div

        return loss, (mse_loss, kl_div)

    def samples(self):
        sample = torch.normal(torch.zeros(self.latent_dim), 1)
        return self.decode(sample)

    def test_encoder_decoder(self, x):
        print(x.shape)
        z = self.encoder(x)
        print(z.shape, self.cnn_output_size)

        d = self.decoder(z)
        print(d.shape)