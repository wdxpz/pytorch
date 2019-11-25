import torch
import torch.nn as nn

from DCGAN.config import DCGAN_Config


class Generator(nn.Module):

    def __init__(self, config = DCGAN_Config):
        super(Generator, self).__init__()
        self.ngpu = config['ngpu']
        self.ngf = config['ngf']
        self.nz = config['nz']
        self.nc = config['nc']
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            # state size. (ngf*8)*4*4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4)*8*8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2)*16*16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf)*32*32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc)*64*64
        )

    def forward(self, *input):
        return self.net(input[0])


class Discriminator(nn.Module):
    def __init__(self, config = DCGAN_Config):
        super(Discriminator, self).__init__()
        self.ngpu = config['ngpu']
        self.ndf = config['ndf']
        self.nc = config['nc']
        self.net = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf*4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #state size (ndf*8)*4*4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            #state size ndf*1*1
            nn.Sigmoid()
        )

    def forward(self, *input):
        x = self.net(input[0])
        return x
