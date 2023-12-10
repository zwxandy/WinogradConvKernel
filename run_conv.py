import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from winograd_kernel import Winograd
from winograd_kernel_origin import Winograd
from winograd_kernel_parall import Winograd_Parall
import time


if __name__ == "__main__":

    """
    Winograd convolution is designed for 3x3 kernel and stride of 1
    """

    input = torch.randn(1, 3, 8, 8)
    weight = torch.randn(6, 3, 3, 3)
    m = 2  # set the size of output tile (optional: 2 or 4)

    # zero padding as neural networks do
    pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
    input = pad(input)

    # regular convolution (correct answer)
    t1 = time.time()
    out_conv2d = F.conv2d(input=input, weight=weight, padding=0)
    t2 = time.time()
    print('Regular convolution:\n', out_conv2d.shape, '\n', 'Time: {:.2f} s'.format(t2-t1), '\n')

    # Winograd convolution w/o parall
    winograd = Winograd(m=m)
    t1 = time.time()
    out_winconv = winograd(input, weight)
    t2 = time.time()
    print('Winograd w/o parall:\n', out_winconv.shape, '\n', 'Time: {:.2f} s'.format(t2-t1))
    if torch.equal(torch.round(out_winconv, decimals=2), torch.round(out_conv2d, decimals=2)):
        print('Correct!\n')
    else:
        print('Wrong!\n')
    
    # Winograd convolution w/ parall
    winograd_parall = Winograd_Parall(m=m)
    t1 = time.time()
    out_winconv_parall = winograd_parall(input, weight)
    t2 = time.time()
    print('Winograd w/ parall:\n', out_winconv_parall.shape, '\n', 'Time: {:.2f} s'.format(t2-t1))
    if torch.equal(torch.round(out_winconv_parall, decimals=2), torch.round(out_conv2d, decimals=2)):
    # if (torch.round(out_winconv_parall, decimals=4) == torch.round(out_conv2d, decimals=4)).all():
        print('Correct!\n')
    else:
        print('Wrong!\n')
