# The definition of the basic SAGAN which generates 64x64 or 128x128 images
# https://arxiv.org/pdf/1805.08318.pdf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.nn.utils import SpectralNorm


def SelfAttention(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.f = nn.Conv2d(n_in, n_in // 8, kernel_size=1, stride=1) # n_filters = n_in //8
        self.g = nn.Conv2d(n_in, n_in //8, kernel_size=1, stride=1))
        self.h = nn.Conv2d(n_in, n_in, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x has dimensions bs x n_in x h x w
        bs, c, _, _ = x.size()
        # apply 1x1 convolutions
        fx = self.f(x) # bs x n_in //8 x h x w
        gx = self.g(x) # bs x n_in //8 x h x w
        hx = self.h(x) # bs x n_in x h x w
        # flatten out the last 2 dimensions
        bs, c, h, w = x.size()
        fx = fx.view(bs, c, -1).permute(0, 2, 1)  # bs x h*w x n_in //8
        gx = gx.view(bs, c, -1) # bs x n_in //8 x h*w
        hx = hx.view(bs, c, -1) # bs x n_in x h*w
        print(fx.shape)
        print(gx.shape)
        print(hx.shape)
        # multiply f and g and apply softmax on each row to get the attention map (bs x h*w x h*w)
        attn = F.softmax(dim=-1)(torch.matmul(fx, gx))
        print('attn {}'.format(attn.shape))
        # out = torch.bmm(proj_value,attention.permute(0,2,1) )
        # in the tf imple: matmul(attn, hx)
        y = torch.matmul(hx, attn) # batch multiply optional attn.permute(0, 2, 1)
        y = y.view(x.size()) # TODO:  check that
        res = self.gamma * y + x
        return y


class DeconvBlock(nn.Module):
    def __init__(self, n_in, n_out, ks, stride, pad, bn=True):
        super().__init__()
        self.conv = SpectralNorm(nn.ConvTranspose2d(n_in, n_out, ks, stride, padding=pad, bias=False))
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.bn(x) if self.bn else x


def conv_block(n_in, n_out, ks, stride, pad=None, bn=True):
    if pad is None:
        pad = ks//2//stride
    if bn == True:
        res = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_in, n_out, kernel_size=ks, bias=False, stride=stride, padding=pad)),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    else:
        res = nn.Sequential(
            SpectralNorm(nn.Conv2d(n_in, n_out, kernel_size=ks, bias=False, stride=stride, padding=pad)),
            nn.BatchNorm2d(n_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    return res

class Discriminator(nn.Module):
    def __init__(self, im_size, ks, ndf, nc=3, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.im_size = im_size
        self.main64 = nn.Sequential(
            # input: nc x 64 x 64
            conv_block(nc, ndf, ks, 2, 1, bn=False), # l1: 3 -> 64
            # ndf x 32 x 32
            conv_block(ndf, ndf*2, ks, 2, 1), # l2: 64 -> 128
            SelfAttention(ndf*2),
            # ndf*2 x 16 x 16
            conv_block(ndf*2, ndf*4, ks, 2, 1), # l3: 128 -> 256
            # ndf*4 x 8 x 8
            conv_block(ndf*4, ndf*8, ks, 2, 1), # l4: 256 -> 512
            # ndf*8 x 4 x 4
            # the last cov has 1 channel and a grid size of no more than 4x4. So we are going to spit out 4x4x1 tensor
            nn.Conv2d(ndf*8, 1, ks, 1, 0, bias=False) # l5: 512 -> 1
        )
        self.main128 = nn.Sequential(
            # input: nc x 128 x 128
            conv_block(nc, ndf, ks, 2, 1, bn=False), # l1: 3 -> 64
            # conv2: ndf x 64 x 64
            conv_block(ndf, ndf*2, ks, 2, 1), #l2:  64 -> 128
            # conv3: ndf*2 x 32 x 32
            conv_block(ndf*2, ndf*4, ks, 2, 1), #l3 128 -> 256
            SelfAttention(ndf*4),
            # conv4: ndf*4 x 16 x 16
            conv_block(ndf*4, ndf*8, ks, 2, 1), # l4: 256 -> 512
            # conv5: ndf*8 x 8 x 8
            conv_block(ndf*8, ndf*16, ks, 2, 1), # l5: 512 -> 1024
            # the last cov has 1 channel and a grid size of no more than 4x4. So we are going to spit out 4x4x1 tensor
            nn.Conv2d(ndf*16, 1, ks, 1, 0, bias=False)
        )

    def forward(self, input):
        if self.im_size == 64:
            return self.main64(input).mean(0).view(1)
        else:
            return self.main128(input).mean(0).view(1)


class Generator(nn.Module):
    def __init__(self, im_size, ks, nz, ngf, nc=3, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        self.im_size = im_size
        self.main64 = nn.Sequential(
            # input z going into convolution
            DeconvBlock(nz, ngf*8, ks, 1, 0), # l1:  nz -> 512
            # ngf*8 x 4 x 4
            DeconvBlock(ngf*8, ngf*4, ks, 2, 1), # l2: 512 -> 256
            # ngf*4 (256) x 8 x 8
            DeconvBlock(ngf*4, ngf*2, ks, 2, 1), # l3: 256 -> 128
            # ngf*2 x 16 x 16
            DeconvBlock(ngf*2, ngf, ks, 2, 1), # l4: 128 -> 64
            SelfAttention(ngf),
            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, ks, 2, 1, bias=False), # l5: 64 -> 3
            nn.Tanh()
            # output: nc x 64 x 64
        )
        self.main128 = nn.Sequential(
            # input z going into convolution
            DeconvBlock(nz, ngf*16, ks, 1, 0), # l1:  nz -> 1024
            # ngf*16 x 4 x 4
            DeconvBlock(ngf*16, ngf*8, ks, 2, 1),  # l2:  1024 -> 512
            # ngf*8 (256) x 8 x 8
            DeconvBlock(ngf*8, ngf*4, ks, 2, 1), # l3: 512 -> 256
            # ngf*4 x 16 x 16
            DeconvBlock(ngf*4, ngf*2, ks, 2, 1), # l4: 256 -> 128
            SelfAttention(ngf*2),
            # ngf*2 x 32 x 32
            DeconvBlock(ngf*2, ngf, ks, 2, 1), # l5: 128 -> 64
            # TODO: SelfAttention(64), optional
            # ngf x 64 x 64 - optional for 128 x128 ims
            nn.ConvTranspose2d(ngf, nc, ks, 2, 1, bias=False), # l6: 64 -> 3
            nn.Tanh()
            # output: nc x 128 x 128
        )
    def forward(self, input):
        if self.im_size == 64:
            return self.main64(input)
        else:
            return self.main128(input)
