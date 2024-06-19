import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.abstract_architecture import AbstractArchitecture
from util.torch_util.torch_util import count_parameters
from util.types import ConfigDict


class UNet(AbstractArchitecture):
    def __init__(self, network_config: ConfigDict, in_features: int, out_features: int, dim: int, use_gpu: bool):
        assert dim in [2, 3], f"Only 2D and 3D U-Nets are supported, given '{dim}'"
        super(UNet, self).__init__(network_config=network_config,
                                   dim=dim,
                                   use_gpu=use_gpu)
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

        depth = network_config.get("depth")
        initial_channels = network_config.get("initial_channels")
        channels = [initial_channels * (2 ** i) for i in range(depth)]

        # Downsampling Path
        for channel in channels:
            self.downs.append(
                nn.Sequential(
                    conv(in_features, channel, kernel_size=3, padding=1),
                    batch_norm(channel),
                    nn.ReLU(inplace=True),
                    conv(channel, channel, kernel_size=3, padding=1),
                    batch_norm(channel),
                    nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
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
                        nn.ReLU(inplace=True)
                    )
                )
            )

        # Final Convolution
        self.final_conv = conv(channels[0], out_features, kernel_size=1)

        training_config = network_config.get("training")
        self._initialize_optimizer_and_scheduler(training_config=training_config)

        # move to device
        self.to(self._gpu_device)

    def forward(self, x):
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

        return self.final_conv(x)


#########
# Tests #
#########

def test_unet_2d(network_config, image_resolution, in_features, out_features, device):
    import matplotlib.pyplot as plt

    def tensor_to_image(tensor):
        """ Convert a PyTorch tensor to a NumPy image. """
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)

    def visualize(input_image, output_image):
        """ Display the input and output images side by side. """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(input_image)
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        ax[1].imshow(output_image, cmap='gray')
        ax[1].set_title('Output Image')
        ax[1].axis('off')

        plt.show()

    # Create a random RGB image
    input_image = torch.randn(1, in_features, image_resolution, image_resolution, device=device)
    # shape (batch_size, in_features, height, width)

    unet = UNet(in_features=in_features, out_features=out_features,
                network_config=network_config, dim=2, use_gpu=True)
    unet.eval()

    # Disable gradient computation for testing (saves memory and computations)
    with torch.no_grad():
        output_image = unet(input_image)

    # Convert tensors to images
    input_np = tensor_to_image(input_image[0])
    output_np = tensor_to_image(output_image[0])

    # Normalize output image to display properly
    output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())

    # Visualize the results
    visualize(input_np, output_np)

    if torch.cuda.is_available():
        from src.hmpn.hmpn_util.calculate_max_batch_size import estimate_max_batch_size
        max_batch_size = estimate_max_batch_size(model=unet, input_sample=input_image, rel_tolerance=0.15, verbose=True)
        print(f"Estimated max batch size: {max_batch_size}")

    total_params = count_parameters(unet)
    print(f"Total trainable parameters: {total_params}")

def test_unet_3d(network_config, image_resolution, in_features, out_features, device):
    # Create a random RGB image
    input_image = torch.randn(1, in_features, 2*image_resolution, image_resolution, image_resolution, device=device)
    # shape (batch_size, in_features, height, width)

    unet = UNet(in_features=in_features, out_features=out_features,
                network_config=network_config, dim=3, use_gpu=True)
    unet.eval()

    with torch.no_grad():
        output_image = unet(input_image)

    print(f"Output shape: {output_image.shape}")

    from src.environments.util.mesh_visualization import get_3d_scatter_trace
    import numpy as np
    import plotly.graph_objects as go
    x = np.linspace(0, 2, 2*image_resolution)
    y = np.linspace(0, 1, image_resolution)
    z = np.linspace(0, 1, image_resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
    scatter_trace = get_3d_scatter_trace(positions=points, scalars=output_image[0, 0].flatten(),
                                         size=3,
                                         colorbar_title="Output Image")
    fig = go.Figure(data=[scatter_trace])
    fig.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Parameters
    image_resolution = 8  # Ensure resolution is suitable for the U-Net (divisible by 16)
    in_features = 3  # RGB image
    out_features = 1

    # Initialize the UNet model
    network_config = {
        "depth": 2,
        "initial_channels": 16,
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1e-3,
        }
    }

    # test_unet_2d(network_config=network_config,
    #              image_resolution=image_resolution,
    #              in_features=in_features,
    #              out_features=out_features,
    #              device=device
    #              )
    #
    test_unet_3d(network_config=network_config,
                 image_resolution=image_resolution,
                 in_features=in_features,
                 out_features=out_features,
                 device=device
                 )


if __name__ == '__main__':
    main()
