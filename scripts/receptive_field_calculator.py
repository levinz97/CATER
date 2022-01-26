import numpy as np

def calculate_rf(kernel_size, dialtion, stride, n_layers):
    print('kernel_size {} dialtion {}, stride {}, n_layers {}'.format(kernel_size, dialtion, stride, n_layers))
    kernel_size = (kernel_size-1)*dialtion + 1
    if isinstance(stride, int):
        stride = np.repeat(stride, n_layers)
    cumprod_stride = np.cumprod(stride)
    print(f'cumprod_stride={cumprod_stride}')
    rf = kernel_size
    for i in range(1,n_layers):
        rf += (kernel_size-1)*cumprod_stride[i-1]
        print(f'layer{i+1} has receptive field {rf}')
    return rf


if __name__ == '__main__':
    n_layers = 5
    print(calculate_rf(kernel_size=3, dialtion=1, stride=2, n_layers=n_layers))
    print(calculate_rf(kernel_size=3, dialtion=2, stride=2, n_layers=n_layers))
    print(calculate_rf(kernel_size=3, dialtion=3, stride=2, n_layers=n_layers))
    print(calculate_rf(kernel_size=3, dialtion=4, stride=2, n_layers=n_layers))
    print(calculate_rf(kernel_size=5, dialtion=2, stride=2, n_layers=n_layers))
    print(calculate_rf(kernel_size=5, dialtion=3, stride=2, n_layers=n_layers))
    


