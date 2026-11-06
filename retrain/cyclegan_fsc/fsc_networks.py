"""CycleGAN generator and discriminator with the fixed smooth convolutional layers of
Kinoshita and Kiya (ICASSP 2020), following Osakabe, Tanaka, Kinoshita, Kiya,
"CycleGAN without checkerboard artifacts for counter-forensics of fake-image detection" (IWAIT 2021).

What changes against the stock networks in networks.py:
  * every stride-2 upsampling in the ResNet generator (ConvTranspose2d) is followed by a fixed,
    non-trainable hold kernel (UpSampling2d, hold_last), which removes the periodic checkerboard
    pattern that transposed convolutions leave in the output (the "model fingerprint" peaks),
  * every stride-2 downsampling in the generator and in the PatchGAN discriminator is preceded by the
    same fixed kernel (DownSampling2d, hold_first), which removes the checkerboard pattern that
    strided convolutions leave in the gradients.
The hold kernel is a zero-order hold (2x2 ones, order 0) or its self-convolution (3x3 triangle,
order 1). Set the order with the environment variable FSC_ORDER (default 0).

Selected from the command line with  --netG resnet_9blocks_fsc  and  --netD basic_fsc .
"""
import functools
import os

import torch.nn as nn

from models.fsc_layer import DownSampling2d, UpSampling2d
from models.networks import ResnetBlock, get_norm_layer

FSC_ORDER = int(os.environ.get("FSC_ORDER", "0"))


class ResnetGeneratorFSC(nn.Module):
    """ResNet generator of networks.py with fixed smooth convolutional layers at every resampling step."""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type="reflect", order=FSC_ORDER):
        assert n_blocks >= 0
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # hold kernel, then the strided convolution
            mult = 2 ** i
            model += [DownSampling2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias,
                                     order=order, hold_mode="hold_first", bias_mode="bias_last"),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # transposed convolution (kernel 4 gives exactly 2x), then the hold kernel
            mult = 2 ** (n_downsampling - i)
            model += [UpSampling2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1, bias=use_bias,
                                   order=order, hold_mode="hold_last", bias_mode="bias_first"),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminatorFSC(nn.Module):
    """PatchGAN discriminator of networks.py with a hold kernel in front of every stride-2 convolution."""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, order=FSC_ORDER):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw, padw = 4, 1
        sequence = [DownSampling2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=True,
                                   order=order, hold_mode="hold_first", bias_mode="bias_last"), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [DownSampling2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias,
                                        order=order, hold_mode="hold_first", bias_mode="bias_last"),
                         norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def build_G(netG, input_nc, output_nc, ngf, norm, use_dropout):
    norm_layer = get_norm_layer(norm_type=norm)
    n_blocks = 9 if netG.startswith("resnet_9blocks") else 6
    return ResnetGeneratorFSC(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks)


def build_D(input_nc, ndf, n_layers_D, norm):
    norm_layer = get_norm_layer(norm_type=norm)
    return NLayerDiscriminatorFSC(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
