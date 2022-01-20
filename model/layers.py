from torch import nn

def conv_bn_relu(input_channels, output_channels, kernel_size, stride=1, dilation=1, padding=0, use_bn=True, use_relu=True):
    layers = []
    layers.append(
        nn.Conv2d(input_channels,
                  output_channels,
                  kernel_size,
                  stride,
                  padding,
                  dilation=dilation,
                  bias=not use_bn)
    )
    if use_bn:
        layers.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        layers.append(nn.LeakyReLU(0.01, inplace=True))

    return nn.Sequential(*layers)

class Decoder(nn.Module):
    def __init__(self, in_channels, n_layers, use_upsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_upsample = use_upsample
        self.n_layers = n_layers
        assert in_channels % (2**n_layers) == 0, f'in_channel = {in_channels} not divisible by {2**(n_layers)}'
        for i in range(n_layers):
            conv = conv_bn_relu(in_channels//(2**i), in_channels//(2**(i+1)), kernel_size=3, padding=1)
            self.add_module(f'decoder_conv{i+1}', conv)
            if use_upsample:
                upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.add_module(f'decoder_upsample{i+1}', upsample)
        

    def forward(self, x):
        for i in range(self.n_layers):
            conv_name = f'decoder_conv_bn_relu{i+1}'
            x = getattr(self, conv_name)(x)
            if self.use_upsample:
                upsample_name = f'decoder_upsample{i+1}'
                x = getattr(self, upsample_name)(x)
        return x