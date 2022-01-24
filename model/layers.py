from detectron2.layers.wrappers import cat
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


class GroupedDilatedConv(nn.Module):
    def __init__(self,input_channels, output_channels, kernel_size, stride=1, dilations=1, padding=0, use_bn=True, use_relu=True):
        super().__init__()
        if isinstance(dilations, int):
            layer_name = self._name_layers(0)
            padding = (kernel_size-1) * dilations // 2
            layer = conv_bn_relu(input_channels, output_channels, kernel_size, stride, dilations, padding, use_bn=use_bn, use_relu=use_relu)
            self.add_module(layer_name, layer)
            self.n_dilations = 1

        elif isinstance(dilations, list):
            self.n_dilations = len(dilations)
            output_channel = output_channels // self.n_dilations
            for i in range(len(dilations)):
                layer_name = self._name_layers(i)
                padding = (kernel_size-1) * dilations[i] // 2
                layer = conv_bn_relu(input_channels, output_channel, kernel_size, stride, dilations[i], padding, use_bn=use_bn, use_relu=use_relu)
                self.add_module(layer_name, layer)
        else:
            raise TypeError("dilations must be a list or a single integer")

    def _name_layers(self, i:int):
        return f'Parallel_conv_{i+1}'

    def forward(self, x):
        output = None
        for i in range(self.n_dilations):
            layer_name = self._name_layers(i)
            y = getattr(self, layer_name)(x)
            if i == 0:
                output = y
            else:
                output = cat([output,y],dim=1)
        return output

class GroupedDilatedConvV2(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, dilations=[1,2,3], padding=0, use_bn=True, use_relu=True):
        super().__init__()
        if isinstance(dilations, int):
            layer_name = self._name_layers(0)
            padding = (kernel_size-1) * dilations // 2
            layer = conv_bn_relu(input_channels, output_channels, kernel_size, stride, dilations, padding, use_bn=use_bn, use_relu=use_relu)
            self.add_module(layer_name, layer)
            self.n_dilations = 1

        elif isinstance(dilations, list):
            self.n_dilations = len(dilations)
            output_channel = input_channels // self.n_dilations
            for i in range(len(dilations)):
                layer_name = self._name_layers(i)
                layers = []
                input_channel = input_channels
                for k, s, d in zip((1,3), (1, stride), (1, dilations[i])):
                    padding = (k-1) * d // 2
                    layer = conv_bn_relu(input_channel, output_channel, k, s, d, padding, use_bn=use_bn, use_relu=use_relu)
                    layers.append(layer)
                    input_channel = output_channel
                self.add_module(layer_name, nn.Sequential(*layers))
            self.last_conv1 = conv_bn_relu(output_channel*self.n_dilations, input_channels, kernel_size=1)
            # self.downsample_conv1 = conv_bn_relu(input_channels, input_channels, kernel_size=1, stride=2)
            self.downsample = nn.MaxPool2d(1,2)
        else:
            raise TypeError("dilations must be a list or a single integer")

    def _name_layers(self, i:int):
        return f'Parallel_conv_{i+1}'

    def forward(self, x):
        output = None
        for i in range(self.n_dilations):
            layer_name = self._name_layers(i)
            y = getattr(self, layer_name)(x)
            if i == 0:
                output = y
            else:
                output = cat([output,y],dim=1)
        output = self.last_conv1(output)
        # x = self.downsample_conv1(x)
        x = self.downsample(x)
        output = cat([output, x], dim=1)
        return output

class Decoder(nn.Module):
    def __init__(self, in_channels, n_layers, use_upsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_upsample = use_upsample
        self.n_layers = n_layers
        assert in_channels % (2**n_layers) == 0, f'in_channel = {in_channels} not divisible by {2**(n_layers)}'
        for i in range(n_layers):
            # apply depth-wise convolution
            conv1 = conv_bn_relu(in_channels//(2**i), in_channels//(2**(i+1)), kernel_size=1)
            self.add_module(f'decoder_conv1_{i+1}', conv1)
            conv3 = conv_bn_relu(in_channels//(2**(i+1)), in_channels//(2**(i+1)), kernel_size=3, padding=1)
            self.add_module(f'decoder_conv3_{i+1}', conv3)
            if use_upsample and (i-1) % 2 == 0:
                upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.add_module(f'decoder_upsample{i+1}', upsample)
        

    def forward(self, x):
        for i in range(self.n_layers):
            conv1_name = f'decoder_conv1_{i+1}'
            x = getattr(self, conv1_name)(x)
            conv3_name = f'decoder_conv3_{i+1}'
            x = getattr(self, conv3_name)(x)
            if self.use_upsample and (i-1) % 2 == 0:
                upsample_name = f'decoder_upsample{i+1}'
                x = getattr(self, upsample_name)(x)
        return x

class SELayer(nn.Module):
    """
    inplement squeeze and excitation layer to learn to reweight the importance of image and backbone features
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(1)
        # use sigmoid instead of relu to create a weight for each channel
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(in_channels//reduction, in_channels),
            nn.Sigmoid() 
        )

    def forward(self, x):
        b, c, _,_ = x.size()
        # squeeze by aggregating feature maps across their spatial dimensions
        y = self.avg_pooling_layer(x).squeeze()
        # excitation, view to enable broadcasting
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)

