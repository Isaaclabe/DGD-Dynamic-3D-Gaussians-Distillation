import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels, latent_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, latent_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(latent_channels, latent_channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(latent_channels * 2, latent_channels * 4, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_channels, output_channels):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_channels * 4, latent_channels * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(latent_channels * 2, latent_channels, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(latent_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x

class AENetwork(nn.Module):
    def __init__(self, image_shape, latent_dim):
        input_channels = (image_shape[0], image_shape[1], image_shape[2])
        latent_channels = (image_shape[0], image_shape[1], latent_dim)
        
        super(AENetwork, self).__init__()
        self.encoder = Encoder(input_channels, latent_channels)
        self.decoder = Decoder(latent_channels, input_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
