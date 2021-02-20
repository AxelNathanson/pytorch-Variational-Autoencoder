import torch
import torch.nn as nn

torch.set_default_dtype(torch.float)

class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 kl_weight: float,
                 image_dim: int = 28,
                 beta: int = 1):
        super().__init__()
        self.beta = beta
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.in_channels = in_channels
        self.image_dim = image_dim

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

    def reparameterization(self, mu, logsigma):
        # This is done to make sure we get a positive semi-definite cov-matrix.
        sigma = torch.exp(logsigma*.5)

        # The reparameterization-trick
        z_tmp = torch.randn_like(sigma)
        z = mu + sigma * z_tmp
        return z

    def forward(self, x):
        mu, logsigma = self.encode(x)

        z = self.reparameterization(mu, logsigma)

        output = self.decode(z).view(-1, self.in_channels, self.image_dim, self.image_dim)

        return output, (mu, logsigma)

    def compute_loss(self, x, output, mu, logsigma):
        # First we compare how well we have recreated the image
        mse_loss = nn.functional.binary_cross_entropy(output.view(x.shape[0], -1),
                                                      x.view(x.shape[0], -1))

        # Then the KL_divergence
        kl_div = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim=1), dim=0)

        loss = mse_loss + self.beta*self.kl_weight*kl_div

        return loss, (mse_loss, kl_div)

    def sample(self, num_samples=1):
        sample = torch.randn(num_samples, self.latent_dim)

        return self.decode(sample)

    def sample_latent_space(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterization(mu, logsigma)
        return z

    def test_encoder_decoder(self, x):
        print(x.shape)
        z = self.encoder(x)
        print(z.shape, self.cnn_output_size)

        d = self.decoder(z)
        print(d.shape)


class fcVAE(BetaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 kl_weight: float,
                 image_dim: int = 28,
                 beta: int = 1):
        super().__init__(in_channels, latent_dim, kl_weight, image_dim, beta)

        self.input_size = in_channels*image_dim**2

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )

        self.linear_mu = nn.Linear(512, self.latent_dim)
        self.linear_logsigma = nn.Linear(512, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_size),
        )

        self.activation = torch.sigmoid

    def encode(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)

        mu = self.linear_mu(z)
        logsigma = self.linear_logsigma(z)

        return mu, logsigma

    def decode(self, x):
        z = self.decoder(x)

        return self.activation(z).view(-1, self.in_channels, self.image_dim, self.image_dim)


class cnnVAE(BetaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 kl_weight: float,
                 image_dim: int = 28,
                 beta: int = 1):
        super().__init__(in_channels, latent_dim, kl_weight, image_dim, beta)

        cnn_channels = [16, 32, 32, 32]
        self.channels_into_decoder = cnn_channels[2]

        # We need two Linear layers to convert encoder -> mu, sigma
        # But first we need to calculate how big the output from our network is.
        self.cnn_output_size = cnn_output_size(image_dim)
        encoder_output_size = cnn_channels[2] * self.cnn_output_size ** 2

        self.linear_mu = nn.Linear(encoder_output_size, self.latent_dim)
        self.linear_logsigma = nn.Linear(encoder_output_size, self.latent_dim)

        self.upsample = nn.Linear(self.latent_dim, encoder_output_size)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[1], out_channels=cnn_channels[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[2], out_channels=cnn_channels[1], kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[1], out_channels=cnn_channels[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(cnn_channels[0], out_channels=cnn_channels[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(cnn_channels[3], out_channels=in_channels, kernel_size=4, padding=1)
        )

        self.activation = torch.sigmoid

    def encode(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)

        mu = self.linear_mu(z)
        logsigma = self.linear_logsigma(z)

        return mu, logsigma

    def decode(self, x):
        z = self.upsample(x).view(-1, self.channels_into_decoder, self.cnn_output_size, self.cnn_output_size)
        z = self.decoder(z)

        return self.activation(z)


########### Extra function ################
def cnn_output_size(input_dim=28, num_channels=3):
    dim = input_dim
    for i in range(num_channels):
        dim = (dim - 3 + 2 * 1) / 2 + 1
    return int(dim)
