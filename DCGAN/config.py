import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DCGAN_Config = {
    'data_root': os.path.join(BASE_DIR, 'data'),
    'workers': 2,
    'ngpu': 2,
    'batch_size': 128,
    'lr': 2e-4,
    'beta1': 0.5,
    'num_epochs': 5,
    'image_size': 64,
    'nz': 100,
    'nc': 3, #image channels
    'ngf': 64, #feature map size in generator
    'ndf': 64, #feature map size in discriminator
}
