import os
import requests
import io
from zipfile import ZipFile

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from DCGAN.config import DCGAN_Config, BASE_DIR
from DCGAN.network import Generator, Discriminator


class DCGAN(object):
    def __init__(self, config=DCGAN_Config):
        self.data_root = config['data_root']
        self.image_size = config['image_size']
        self.workers = config['workers']
        self.nz = config['nz']
        self.num_epochs = config['num_epochs']
        self.lr = config['lr']
        self.beta1 = config['beta1']
        self.batch_size = config['batch_size']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.real_data_loader = self._get_real_data()
        self.generator, self.discriminator = self._init_network()

    def _get_real_data(self):
        filename = os.path.join(self.data_root, 'img_align_celeba.zip')
        if not os.path.exists(filename):
            # url = 'https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg'
            # downloaded = requests.get(url)
            # open(filename, 'wb').write(downloaded.content)
            raise Exception('img_align_celeba.zip not existed! '
                            'please download from https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg')
        else:
            print('{} already existed.'.format('img_align_celeba.zip'))

        dirname = os.path.join(self.data_root, 'img_align_celeba')
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            with ZipFile(filename, 'r') as zipObj:
                zipObj.extractall(dirname)

        real_dataset = datasets.ImageFolder(dirname,
                                            transform=transforms.Compose([
                                                transforms.Resize(self.image_size),
                                                transforms.CenterCrop(self.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        data_loader = torch.utils.data.DataLoader(real_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.workers)

        return data_loader

    def _init_network(self):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            if type(m) == nn.BatchNorm2d:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        generator = Generator(DCGAN_Config).to(self.device)
        generator.apply(init_weights)

        discriminator = Discriminator(DCGAN_Config).to(self.device)
        discriminator.apply(init_weights)

        if self.device.type =='cuda' and DCGAN_Config['ngpu']>1:
            generator = nn.DataParallel(generator, list(range(DCGAN_Config['ngpu'])))
            discriminator = nn.DataParallel(discriminator, list(range(DCGAN_Config['ngpu'])))

        print(generator)
        print(discriminator)

        return generator, discriminator

    def train(self):
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        criterion = nn.BCELoss()

        #create batch of letent vector that we will use to visualize the progression of the generator
        fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        real_label = 1
        fake_label = 0

        #set up optimizers
        optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        print('Starting Training Loop...')
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.real_data_loader, 0):
                self.discriminator.zero_grad()

                real_data = data[0].to(self.device)
                b_size = real_data.size(0)


                #################################################################
                # (1) Update D network: maximize y*log(D(x)) + (1-y)log(1-D(G(z)))
                #################################################################
                ##train with real batch
                #forward and calculate discriminator loss on real data
                output = self.discriminator(real_data).view(-1)
                label = torch.full((b_size,), real_label, device=self.device)
                errD_real = criterion(output, label)
                #calculate gradients for D in backdward pass
                errD_real.backward()
                D_x = output.mean().item()

                ##train with fake batch
                #generate fake batch
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake_data = self.generator(noise)
                label.fill_(fake_label)
                # forward and calculate discriminator loss on real data
                output = self.discriminator(fake_data).view(-1)
                errD_fake = criterion(output, label)
                # calculate gradients for D in backdward pass
                errD_fake.backward(retain_graph=True)
                D_G_z1 = output.mean().item()

                #add the gradients from real batch and fack batch, and update D
                errD = errD_real + errD_fake
                optimizerD.step()

                #################################################################
                # (2) Update G network: maximize log(D(G(z)))
                #################################################################
                self.generator.zero_grad()

                #directly use fake batch in step (1), and change lable as real to calculate log(D(G(z)))
                label.fill_(real_label)
                output = self.discriminator(fake_data).view(-1)
                errG = criterion(output, label)
                #backward and update G
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                #Output training stats
                if (i%50) == 0:
                    print('epoch:[{}/{}] batch:[{}/{}]\t Loss_D: {:.4f}\t Loss_G: {:.4f}\tD(x): {:.4f}\t'
                          'D(G(z)): {:.4f} / {:.4f}'.format(
                        epoch, self.num_epochs, i, len(self.real_data_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2
                    ))

                #save losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())


                #check how the generator is doing by save G's output on fixed noise
                if (iters%500 ==0) or ((epoch==self.num_epochs-1) and (i == len(self.real_data_loader)-1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1


        img_dir = os.path.join(BASE_DIR, 'image')
        for i in range(len(img_list)):
            imgname = os.path.join(img_dir, 'image{:03d}.jpg'.format(i))
            vutils.save_image(img_list[i], imgname)


        plt.figure(figsize=(10,5))
        plt.title('Generator and Discriminator Loss During Training')
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label='D')
        plt.xlabel('iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        #%%capture
        fig = plt.figure(figsize=(8,8))
        plt.axis('off')
        ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        HTML(ani.to_jshtml())












