import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, List

from nets.base import BaseVAE


def weights_init(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s type" % init_type)
    net.apply(init_func)


class VanillaVAE(BaseVAE):
    def __init__(
        self,
        device,
        in_channels: int,
        latent_dim: int,
        hidden_dims: List = None,
        **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.device = device
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 16, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 16)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    # nn.UpsamplingBilinear2d(scale_factor=2),
                    # nn.Conv2d(
                    #     hidden_dims[i],
                    #     out_channels=hidden_dims[i + 1],
                    #     kernel_size=3,
                    #     stride=1,
                    #     padding=1,
                    # ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)  # torch.Size([2, 128]) torch.Size([2, 128])
        z = self.reparameterize(mu, log_var)  # torch.Size([2, 128])
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


if __name__ == "__main__":
    in_channels = 3
    latent_dim = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    # 模拟训练过程
    inputs = torch.rand((2, 3, 128, 128)).to(device)
    net = VanillaVAE(device, in_channels, latent_dim)
    net = net.to(device)
    reconstructed_x, input, mu, log_var = net(inputs)
    print(reconstructed_x.shape, mu.shape, log_var.shape)

    # 模拟生成样本过程
    num_samples = 3
    samples = net.sample(num_samples, device)
    print(samples.shape)

    # 模拟计算损失函数
    loss = net.loss_function(reconstructed_x, input, mu, log_var, M_N=0.00025)
    print(loss)
