from torch import nn


def conv_bn_relu(input_channels, output_channels, kernel_size, stride=1, padding=0, use_bn=True, use_relu=True):
    layers = []
    layers.append(
        nn.Conv2d(input_channels,
                  output_channels,
                  kernel_size,
                  stride,
                  padding,
                  bias=not use_bn)
    )
    if use_bn:
        layers.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        layers.append(nn.LeakyReLU(0.01, inplace=True))

    return layers
