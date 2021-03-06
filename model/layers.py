from collections import OrderedDict
from detectron2.layers.wrappers import cat
from torch import nn
import torch.nn.functional as F
# import torch
# torch.autograd.set_detect_anomaly(True)


def conv_bn_relu(input_channels, output_channels, kernel_size, stride=1, dilation=1, padding=0, use_bn=True, use_relu=True, groups=1, inplace=True):
    layers = []
    layers.append(
        nn.Conv2d(input_channels,
                  output_channels,
                  kernel_size,
                  stride,
                  padding,
                  dilation=dilation,
                  bias=not use_bn,
                  groups=groups,)
    )
    if use_bn:
        layers.append(nn.BatchNorm2d(output_channels))
    if use_relu:
        layers.append(nn.LeakyReLU(0.01, inplace=inplace))

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
            self.last_conv1 = conv_bn_relu(output_channel*self.n_dilations, output_channels, kernel_size=1, use_bn=False, use_relu=False)
            self.bn0 = nn.BatchNorm2d(output_channels)
            self.downsample = nn.Sequential()
            if stride != 1 or input_channels != output_channels:
                if stride == 2 and input_channels == output_channels:
                    self.downsample = nn.MaxPool2d(1,2)
                else:
                    self.downsample = conv_bn_relu(input_channels, output_channels, kernel_size=1, stride=stride)
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
        # output = cat([output, x], dim=1)
        output += x
        output = F.relu(self.bn0(output), inplace=True)
        return output

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encode_layers = self._make_layers(in_channels)
    
    def _make_layers(self, in_channels):
        layers = OrderedDict([
            ("ConvBnR_7_0", conv_bn_relu(in_channels, 2*in_channels, kernel_size=7, stride=2, padding=3)),
            ("ResNextBlock_3_1", DilatedResNextBlock(2*in_channels, bottleneck_width=2*in_channels//4, cardinality=4, stride=2, expansion=2)),
            ("ResNextBlock_3_2", DilatedResNextBlock(4*in_channels, bottleneck_width=4*in_channels//4, cardinality=4, stride=2, expansion=2)),
            ("ResNextBlock_3_3", DilatedResNextBlock(8*in_channels, bottleneck_width=8*in_channels//4, cardinality=4, stride=2, expansion=1)),
            ("ResNextBlock_3_4", DilatedResNextBlock(8*in_channels, bottleneck_width=8*in_channels//4, cardinality=4, stride=1, expansion=1)),
        ])

        return nn.Sequential(layers)
    
    def forward(self,x):
        return self.encode_layers(x)

class Encoder_V2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encode_layers = self._make_layers(in_channels)
    
    def _make_layers(self, in_channels):
        layers = OrderedDict([
            ("ConvBnR_7_0", conv_bn_relu(in_channels, 2*in_channels, kernel_size=7, stride=2, padding=3)),
            # ("GroupedDilatedConvV2_3_1", GroupedDilatedConvV2(2*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[6,12,18,24])),
            # ("GroupedDilatedConvV2_3_2", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[1,6,12,18])),
            # ("GroupedDilatedConvV2_3_3", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[1,6,12,18])),
            # ("GroupedDilatedConvV2_3_4", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=1, dilations=[1,6,12,18])),
            ("GroupedDilatedConvV2_3_1", GroupedDilatedConvV2(2*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[6,12,18])),
            ("GroupedDilatedConvV2_3_2", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[1,6,12])),
            ("GroupedDilatedConvV2_3_3", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=2, dilations=[1,6,12])),
            ("GroupedDilatedConvV2_3_4", GroupedDilatedConvV2(4*in_channels, 4*in_channels, kernel_size=3, stride=1, dilations=[1,6,12])),
        ])

        return nn.Sequential(layers)
    
    def forward(self,x):
        return self.encode_layers(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, n_layers, use_upsample=True):
        super().__init__()
        self.in_channels = in_channels
        self.use_upsample = use_upsample
        self.n_layers = n_layers
        assert in_channels % (2**n_layers) == 0, f'in_channel = {in_channels} not divisible by {2**(n_layers)}'
        for i in range(n_layers):
            # apply 1x1 kernel to decrease the number of channels, then followed by 3x3 kernel
            conv1 = conv_bn_relu(in_channels//(2**i), in_channels//(2**(i+1)), kernel_size=1)
            self.add_module(f'decoder_conv1_{i+1}', conv1)
            channels = in_channels//(2**(i+1))
            conv3 = conv_bn_relu(channels, channels, kernel_size=3, padding=1)
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

class DilatedResNextBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_width=7, cardinality=32, stride=1, expansion=2, dilation=2):
        super().__init__()
        inner_width = bottleneck_width * cardinality
        self.expansion = expansion
        padding = (3-1) * dilation // 2
        self.basic = nn.Sequential(OrderedDict(
            [
                ('ConvBnR1_0', conv_bn_relu(in_channels, inner_width, kernel_size=1, stride=1, inplace=False)),
                ('ConvBnR3_0', conv_bn_relu(inner_width, inner_width, kernel_size=3, dilation=dilation, padding=padding, stride=stride, groups=cardinality, inplace=False)),
                ('ConvBnR1_1', conv_bn_relu(inner_width, inner_width*self.expansion, kernel_size=1, stride=1, use_relu=False, inplace=False))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != inner_width*self.expansion:
            self.shortcut = nn.Sequential(OrderedDict([
                ('ConvBnR1_2', conv_bn_relu(in_channels, inner_width*self.expansion, kernel_size=1, stride=stride, use_bn=False, use_relu=False))
            ]))
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)
    
    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out


class SELayer(nn.Module):
    """
    inplement squeeze and excitation layer to reweight image and backbone features
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

