import argparse
import os
import numpy as np
import math
from math import log10
import itertools
import sys
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

import pytorch_ssim

from models import *
from dataset import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()

parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--val_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--jpeg_quality', type=int, default=40)
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--lr_d", type=float, default=0.00001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--experiment', default='argan', type=str)
parser.add_argument('--pre_epoch', default=10, type=int)
parser.add_argument('--perceptual_loss', default=0.006, type=float)
parser.add_argument('--gan_loss_wt', default=0.001, type=float)

mean = [0.3572, 0.3610, 0.3702]
std = [0.2033, 0.2016, 0.2011]
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

opt = parser.parse_args()
opt.experiment = opt.experiment + f"-patch_{opt.patch_size}-lr_g{opt.lr_g}-lr_d{opt.lr_d}-quality_{opt.jpeg_quality}"
print(opt.experiment)

if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

os.makedirs(os.path.join(opt.output_dir, opt.experiment, "images"), exist_ok=True)
os.makedirs(os.path.join(opt.output_dir, opt.experiment, "weights"), exist_ok=True)

if not os.path.exists(os.path.join(opt.output_dir, opt.experiment)):
    os.makedirs(os.path.join(opt.output_dir, opt.experiment))

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

#hr_shape = (opt.patch_size, opt.patch_size)

transforms_train = transforms.Compose([
        transforms.TenCrop(opt.patch_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, std)(crop) for crop in crops])),
    ])

transforms_evaluation = transforms.Compose([
    transforms.CenterCrop(opt.patch_size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(3, opt.patch_size, opt.patch_size))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(os.path.join(opt.output_dir, f"generator_{opt.epoch}.pth")))
    discriminator.load_state_dict(torch.load(os.path.join(opt.output_dir, f"discriminator_{opt.epoch}.pth")))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

train_ds = Dataset(opt.train_dir, opt.patch_size, opt.jpeg_quality, transforms=transforms_train, train=True)
val_ds = Dataset(opt.val_dir, opt.patch_size*3, opt.jpeg_quality, transforms=transforms_train, train=False)
train_eval_ds = Dataset(opt.train_dir, opt.patch_size*3, opt.jpeg_quality, transforms=transforms_train, train=False)

trainloader = DataLoader(dataset=train_ds,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.threads,
                        pin_memory=True,
                        drop_last=True)

valloader = DataLoader(dataset=val_ds,
                        batch_size=opt.batch_size,
                        shuffle=True,
                        num_workers=opt.threads,
                        pin_memory=True,
                        drop_last=True)

trainevalloader = DataLoader(dataset=val_ds,
                        batch_size=opt.batch_size,
                        shuffle=False,
                        num_workers=opt.threads,
                        pin_memory=True,
                        drop_last=True)

print("Data loaded succesfully")

writer = SummaryWriter(os.path.join(opt.output_dir, opt.experiment))

# ----------
#  Training
# ----------

results = {'loss_D': [],
           'loss_D_real': [],
           'loss_D_fake': [],
           'loss_G': [],
           'score_D': [],
           'score_G': [],
           'psnr': [],
           'ssim': []}

#---------------------------
# Pre-training for generator
#---------------------------

for pre_epoch in range(opt.pre_epoch):
    train_bar = tqdm(trainloader)
    for i, imgs in enumerate(train_bar):

            # Configure model input
            imgs_lr = Variable(imgs[0].type(Tensor))
            imgs_hr = Variable(imgs[1].type(Tensor))
            bs, ncrops, c, h, w = imgs_lr.size()
            imgs_lr, imgs_hr = imgs_lr.view(-1, c, h, w), imgs_hr.view(-1, c, h, w)

            imgs_lr = imgs_lr.to(device)
            imgs_hr = imgs_hr.to(device)

            batch_size = opt.batch_size * ncrops


            # ------------------
            #  Train Generator
            # ------------------

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_hr, imgs_hr) + opt.perceptual_loss * criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content

            loss_G.backward()
            optimizer_G.step()


for epoch in range(opt.epoch, opt.n_epochs):
    train_bar = tqdm(trainloader)
    running_results = {'batch_sizes': 0,
                       'loss_G': 0,
                       'loss_D': 0,
                       'loss_D_real': 0,
                       'loss_D_fake': 0,
                       'score_G': 0,
                       'score_D': 0,
                      'l2':0,
                      'ssims':0,
                      'psnr':0,
                      'ssim':0,
                      'ssim_gt':0,
                      'ssims_gt':0,
                       'psnr_gt':0,
                       'l2_gt':0
                      }

    generator.train()
    discriminator.train()

    for i, imgs in enumerate(train_bar):

        # Configure model input
        imgs_lr = Variable(imgs[0].type(Tensor))
        imgs_hr = Variable(imgs[1].type(Tensor))
        bs, ncrops, c, h, w = imgs_lr.size()
        imgs_lr, imgs_hr = imgs_lr.view(-1, c, h, w), imgs_hr.view(-1, c, h, w)
        #print(torch.min(inputs), torch.max(inputs), torch.min(labels), torch.min(labels))

        imgs_lr = imgs_lr.to(device)
        imgs_hr = imgs_hr.to(device)

        batch_size = opt.batch_size * ncrops
        running_results['batch_sizes'] += batch_size
        #tqdm.write(str(imgs_lr.size()) + str(imgs_hr.size()))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generator
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        #tqdm.write(str(criterion_content(gen_hr, imgs_hr))+ " " + str(criterion_content(gen_features,real_features.detach())))
        loss_content = criterion_content(gen_hr, imgs_hr) + opt.perceptual_loss * criterion_content(gen_features, real_features.detach())


        # Total loss
        loss_G = loss_content + opt.gan_loss_wt * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        out_real = discriminator(imgs_hr)
        out_fake = discriminator(gen_hr.detach())

        # Loss of real and fake images
        loss_D_real = criterion_GAN(out_real, valid)
        loss_D_fake = criterion_GAN(out_fake, fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # loss for current batch before optimization
        loss_G_mse_batch = ((gen_hr - imgs_hr)**2).data.mean()
        loss_G_mse_batch_gt = ((imgs_lr - imgs_hr)**2).data.mean()

        running_results['l2'] += loss_G_mse_batch * batch_size
        running_results['l2_gt'] += loss_G_mse_batch_gt * batch_size
        running_results['loss_G'] += loss_G.item() * batch_size
        running_results['loss_D'] += loss_D.item() * batch_size
        running_results['loss_D_real'] += loss_D_real.item() * batch_size
        running_results['loss_D_fake'] += loss_D_fake.item() * batch_size
        running_results['score_D'] += out_real.mean().item() * batch_size
        running_results['score_G'] += out_fake.mean().item() * batch_size

        batch_ssim = pytorch_ssim.ssim(gen_hr, imgs_hr).item()
        batch_ssim_gt = pytorch_ssim.ssim(imgs_lr, imgs_hr).item()

        running_results['ssims_gt'] += batch_ssim_gt * batch_size
        running_results['ssims'] += batch_ssim * batch_size
        running_results['ssim'] = running_results['ssims'] / running_results['batch_sizes']
        running_results["ssim_gt"] = running_results['ssims_gt'] / running_results['batch_sizes']

        running_results['psnr'] = 10 * log10(imgs_hr.max()**2 / (running_results['l2'] / running_results['batch_sizes']))
        running_results['psnr_gt'] = 10 * log10(imgs_hr.max()**2 / (running_results['l2_gt'] / running_results['batch_sizes']))


        train_bar.set_description(desc='[%d/%d] Loss_D: %.3f Loss_G: %.3f D(x): %.3f D(G(z)): %.4f' % (
                epoch, opt.n_epochs, running_results['loss_D'] / running_results['batch_sizes'],
                running_results['loss_G'] / running_results['batch_sizes'],
                running_results['score_D'] / running_results['batch_sizes'],
                running_results['score_G'] / running_results['batch_sizes']))
        train_bar.update()

    train_results = {'loss_G': running_results['loss_G'] / running_results['batch_sizes'],
                     'loss_D': running_results['loss_D'] / running_results['batch_sizes'],
                     'loss_D_real': running_results['loss_D_real'] / running_results['batch_sizes'],
                     'loss_D_fake': running_results['loss_D_fake'] / running_results['batch_sizes'],
                     'score_G': running_results['score_G'] / running_results['batch_sizes'],
                     'score_D': running_results['score_D'] / running_results['batch_sizes'],
                     'psnr': running_results['psnr'],
                     'ssim': running_results['ssim'],
                     'ssim_gt': running_results['ssim_gt'],
                     'psnr_gt': running_results['psnr_gt'],
                    }

    torch.save(generator.state_dict(), os.path.join(opt.output_dir, opt.experiment, "weights", f"generator_{epoch}.pth"))
    torch.save(discriminator.state_dict(),os.path.join(opt.output_dir, opt.experiment, "weights", f"discriminator_{opt.epoch}.pth"))

    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        val_bar = tqdm(valloader)
        val_results = {'l1': 0, 'l2':0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0, 'ssims_gt':0, 'ssim_gt':0,
                       'psnr_gt':0, 'l2_gt':0}
        val_images = []

        for i, imgs in enumerate(val_bar):

            lr = imgs[0]
            hr = imgs[1]
            bs, ncrops, c, h, w = lr.size()
            lr, hr = lr.view(-1, c, h, w), hr.view(-1, c, h, w)
            batch_size = opt.batch_size * ncrops
            val_results['batch_sizes'] += batch_size

            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            gen_hr = generator(lr)

            loss_G_batch = criterion_content(gen_hr, hr)
            loss_G_mse_batch = ((gen_hr - hr)**2).data.mean()
            loss_G_mse_batch_gt = ((lr - hr)**2).data.mean()

            val_results['l2'] += loss_G_mse_batch * batch_size
            val_results['l2_gt'] += loss_G_mse_batch_gt * batch_size

            batch_ssim = pytorch_ssim.ssim(gen_hr, hr).item()
            batch_ssim_gt = pytorch_ssim.ssim(lr, hr).item()

            val_results['ssims_gt'] += batch_ssim_gt * batch_size
            val_results['ssims'] += batch_ssim * batch_size
            val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']
            val_results["ssim_gt"] = val_results['ssims_gt'] / val_results['batch_sizes']

            val_results['psnr'] = 10 * log10(hr.max()**2 / (val_results['l2'] / val_results['batch_sizes']))
            val_results['psnr_gt'] = 10 * log10(hr.max()**2 / (val_results['l2_gt'] / val_results['batch_sizes']))



        val_bar.set_description(desc='PSNR: %.4f dB SSIM: %.4f\n' % (val_results['psnr'], val_results['ssim']))
        val_bar.update()
        val_results = {
                     'psnr': val_results['psnr'],
                     'ssim': val_results['ssim'],
                    'psnr_gt': val_results['psnr_gt'],
                     'ssim_gt': val_results['ssim_gt'],

                    }

        # Save image grid with upsampled inputs and SRGAN outputs
        n = len(gen_hr)
        r = random.sample(list(range(n)), 16)
        gen_hr = make_grid(gen_hr[r], nrow=1, normalize=True)
        imgs_lr = make_grid(lr[r], nrow=1, normalize=True)
        imgs_hr = make_grid(hr[r], nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, imgs_hr, gen_hr), -1)
        save_image(img_grid, os.path.join(opt.output_dir, opt.experiment, "images", "%d.ppm" % epoch), normalize=False)
        tqdm.write("Saved image outputs")

        print("PSNR", train_results["psnr"], val_results["psnr"], train_results["psnr_gt"])
        writer.add_scalars(opt.experiment+"/loss", {'G':train_results["loss_G"], 'D': train_results["loss_D"]}, epoch)
        writer.add_scalars(opt.experiment+"/psnr", {'Train':train_results["psnr"], 'Val': val_results["psnr"], 'JPEG_Val':val_results["psnr_gt"], 'JPEG_Train':train_results["psnr_gt"]}, epoch)
        writer.add_scalars(opt.experiment+"/ssim", {'Train':train_results["ssim"], 'Val': val_results["ssim"], 'JPEG_Val':val_results["ssim_gt"], 'JPEG_Train':train_results["ssim_gt"]}, epoch)

