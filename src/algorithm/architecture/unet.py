import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn


class UNet(nn.Module):
    def __init__(self, architecture_config: DictConfig, example_image: torch.Tensor, out_features: int = 1):
        """

        Args:
            architecture_config:
            example_image: Example torch.Tensor representing a batch of images with a single entry. Has shape
                (batch_size, channels, height, width) for 2D or (batch_size, channels, depth, height, width) for 3D.
            out_features: How many dimensions the output should have per predicted pixel. Default is 1.
        """
        dim = len(example_image.shape) - 2
        in_features = example_image.shape[1]

        assert dim in [2, 3], f"Only 2D and 3D U-Nets are supported, given '{dim}'"
        # calling super with all arguments saves these as kwargs for the save and load
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        if dim == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
            batch_norm = nn.BatchNorm2d
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            conv = nn.Conv3d
            conv_transpose = nn.ConvTranspose3d
            batch_norm = nn.BatchNorm3d
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        depth = architecture_config.depth
        initial_channels = architecture_config.initial_channels
        channels = [initial_channels * (2**i) for i in range(depth)]
        self.min_size = 2**depth

        # Downsampling Path
        for channel in channels:
            self.downs.append(
                nn.Sequential(
                    conv(in_features, channel, kernel_size=3, padding=1),
                    batch_norm(channel),
                    nn.ReLU(inplace=True),
                    conv(channel, channel, kernel_size=3, padding=1),
                    batch_norm(channel),
                    nn.ReLU(inplace=True),
                )
            )
            in_features = channel

        # Bottleneck
        self.bottleneck = nn.Sequential(
            conv(channels[-1], channels[-1] * 2, kernel_size=3, padding=1),
            batch_norm(channels[-1] * 2),
            nn.ReLU(inplace=True),
            conv(channels[-1] * 2, channels[-1] * 2, kernel_size=3, padding=1),
            batch_norm(channels[-1] * 2),
            nn.ReLU(inplace=True),
        )

        # Upsampling Path
        for channel in reversed(channels):
            self.ups.append(
                nn.Sequential(
                    conv_transpose(channel * 2, channel, kernel_size=2, stride=2),
                    nn.Sequential(
                        conv(channel * 2, channel, kernel_size=3, padding=1),
                        batch_norm(channel),
                        nn.ReLU(inplace=True),
                        conv(channel, channel, kernel_size=3, padding=1),
                        batch_norm(channel),
                        nn.ReLU(inplace=True),
                    ),
                )
            )

        # Final Convolution
        self.final_conv = conv(channels[0], out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if the image is too small, pad it to the minimum size
        old_shape = torch.tensor(x.shape[2:], device=x.device)
        slices = [slice(None), slice(None)] + [slice(0, s) for s in old_shape]
        if any(old_shape < self.min_size):
            image_dimension = torch.clamp(old_shape, min=self.min_size)
            x_ = torch.zeros((*x.shape[:2], *image_dimension), device=x.device)

            x_[slices] = x
            x = x_

        # do the forward pass
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx][0](x)
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx][1](x)

        x = self.final_conv(x)

        x = x[slices]  # "unpad" back to original image

        return x
