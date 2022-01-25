import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from layers import GroupedDilatedConv, GroupedDilatedConvV2, Decoder

with torch.cuda.device(0):
#   net = models.densenet161()
    net = models.resnet50()
    in_channels = 122
    # net = GroupedDilatedConv(in_channels, 2*in_channels, 3, 2)
    # net = Decoder(256,5)

    macs, params = get_model_complexity_info(net, (3, 128, 128), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))