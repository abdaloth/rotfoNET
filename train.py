import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils.CUBDataset import CUBDataset, image_dim
from utils.CUBDataset_with_related_captions import CUBDataset_With_Related_Captions
from utils.CUBDataset_with_related_captions import image_dim as rc_image_dim

import models

from torch.utils.data import DataLoader

from tqdm import tqdm
import time
from datetime import datetime
from glob import glob
from argparse import ArgumentParser
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt


# NOTE: hardcoded path, fix
image_filenames = glob("../CUB_200_2011/images/*/*.jpg")

tr = transforms.Compose([
        transforms.Resize((image_dim,image_dim), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])


def PSNRLoss(original, inpainted): 
    eps = 1e-11
    mse = torch.mean((original - inpainted)**2)
    psnr = 20 * torch.log10(1/torch.sqrt(mse+eps)) 
    return -psnr


def show_checkpoint_repros(model):
    # NOTE: hardcoded path, fix
    birb = Image.open("../CUB_200_2011/Vermilion_Flycatcher_0042_42266.jpg") # picture of a room from the validation set.
    caption = "a small bird with a large head, orange abdomen and crown."
    birb = birb.convert("RGB")

    tr =transforms.Compose([
            transforms.Resize((image_dim,image_dim), Image.BICUBIC),
            transforms.ToTensor()
        ])
    birb_t = tr(birb)
    x1 = 112-43
    y1 = 112-43
    x2 = x1 + 86
    y2 = y1 + 86

    mask = torch.zeros(3, image_dim,image_dim,dtype=torch.bool)
    mask[:,y1:y2,x1:x2] = True
    
    masked = birb_t.detach().clone()
    masked[mask] = 1

    with torch.no_grad():
        model.eval()
        recon_full = model(birb_t.unsqueeze(0).cuda(), caption)
        recon_masked = model(masked.unsqueeze(0).cuda(), caption)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(15,5))
        plt.suptitle(caption)
        ax1.imshow(birb_t.permute(1,2,0))
        ax1.set_title("full")
        ax2.imshow(masked.permute(1,2,0))
        ax2.set_title("masked")
        ax3.imshow(recon_full[0].cpu().permute(1,2,0))
        ax3.set_title("reconstructed full")
        ax4.imshow(recon_masked[0].cpu().permute(1,2,0))
        ax4.set_title("reconstructed masked")
        fig.tight_layout()
        plt.show()

def show_checkpoint_repros_similar_images(model):
    # NOTE: hardcoded path, fix
    birb = Image.open("../../data/Vermilion_Flycatcher_0042_42266.jpg") # picture of a room from the validation set.
    caption = "this bird has a red crown and flank as well as a black pointed bill and black tarsus." # I wrote a caption for it
    birb = birb.convert("RGB")
    birb = birb.resize((224,224))

    tr = transforms.ToTensor()
    birb_t = tr(birb)
    x1 = 112-43
    y1 = 112-43
    x2 = x1 + 86
    y2 = y1 + 86

    mask = torch.zeros(3, image_dim,image_dim,dtype=torch.bool)
    mask[:,y1:y2,x1:x2] = True
    
    masked = birb_t.detach().clone()
    masked[mask] = 1
    
    similar_image_indices = np.asarray([92610, 22361, 47458]) // 10

    similar_images = []
    for i in similar_image_indices:
        im = Image.open(image_filenames[i])
        im = im.convert("RGB") # The occasional 1 channel grayscale image is in there.
        im = tr(im)
        similar_images.append(im)

    similars = torch.cat(similar_images) # It's just 3 * num_related channels, e.g. 9x224x224

    with torch.no_grad():
        model.eval()
        model_input = torch.cat([masked.unsqueeze(0), similars.unsqueeze(0)], dim=1)
        recon_masked = model(model_input.cuda())

        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6, figsize=(18,4))
        plt.suptitle(caption)
        ax1.imshow(birb_t.permute(1,2,0))
        ax1.set_title("original")
        ax2.imshow(masked.permute(1,2,0))
        ax2.set_title("masked")
        ax3.imshow(similar_images[0].cpu().permute(1,2,0))
        ax3.set_title("similar image 1")
        ax4.imshow(similar_images[1].cpu().permute(1,2,0))
        ax4.set_title("similar image 2")
        ax5.imshow(similar_images[2].cpu().permute(1,2,0))
        ax5.set_title("similar image 3")
        ax6.imshow(recon_masked[0].cpu().permute(1,2,0))
        ax6.set_title("reconstructed")
        fig.tight_layout()
        plt.show()

def save_sample(batches_done):
    # NOTE: hardcoded path, fix
    birb = Image.open("../CUB_200_2011/Vermilion_Flycatcher_0042_42266.jpg") # picture of a room from the validation set.
    birb = birb.convert("RGB")

    similar_image_indices = np.asarray([92610, 22361, 47458]) // 10

    similar_images = []
    for i in similar_image_indices:
        im = Image.open(image_filenames[i])
        im = im.convert("RGB") # The occasional 1 channel grayscale image is in there.
        im = tr(im)
        similar_images.append(im)

    similars = torch.cat(similar_images).cuda() # It's just 3 * num_related channels, e.g. 9x224x224

    birb_t = tr(birb).unsqueeze(0)
    x1 = 64-32
    y1 = 64-32
    x2 = x1 + 64
    y2 = y1 + 64

    mask = torch.zeros(1, 3, image_dim,image_dim,dtype=torch.bool)
    mask[:,:,y1:y2,x1:x2] = True
    
    masked = birb_t.detach().clone().cuda()
    masked[mask] = 1.0
    model_input = torch.cat([masked, similars.unsqueeze(0)], dim=1)
    # NOTE: we discard the `Variable` encapsulation
    # Generate inpainted image
    gen_mask = generator(model_input)
    masked = masked.cpu()
    filled_samples = masked.clone()
    filled_samples[:, :, y1 : y2, x1 : x2] = gen_mask.cpu()
    # Save sample
    sample = torch.cat((masked.data, filled_samples.data, birb_t.data), -2)

    save_image(sample, "images/%d.png" % batches_done, nrow=1, normalize=True)


# ----------
#  Training
# ----------

def train_gan(epochs, trainloader, batch_size, sample_interval, 
            generator, discriminator, checkpoint_pth):
    # GAN code reference: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/context_encoder
    patch_h, patch_w = int(64 / 2 ** 3), int(64 / 2 ** 3)
    patch = (1, patch_h, patch_w)
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(.5, .999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(.5, .999))

    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()

    Tensor = torch.cuda.FloatTensor

    for epoch in range(epochs):
        for i, batch in enumerate(trainloader):
            imgs = batch['full_images'].cuda()
            similar_images = batch['similar_images'].cuda()
            mask = batch['masks']
            masked_imgs = imgs.detach().clone().cuda()
            masked_imgs[mask] = 1.0
            masked_parts = imgs[mask].view(imgs.size(0), 3, 64, 64)
            
            model_input = torch.cat([masked_imgs, similar_images], dim=1)

            
            # Adversarial ground truths
            valid = Tensor(imgs.shape[0], *patch).fill_(1.0)
            valid.requires_grad=False
            fake = Tensor(imgs.shape[0], *patch).fill_(0.0)
            fake.requires_grad=False

            # Configure input
            imgs = imgs.type(Tensor)
            masked_imgs = masked_imgs.type(Tensor)
            masked_parts =masked_parts.type(Tensor)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(model_input)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()


            # Generate sample at sample interval
            batches_done = epoch * len(trainloader) + i
            if batches_done % 500 == 0:
                save_sample(batches_done)
                print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                % (epoch, 20, i, len(trainloader), d_loss.item(), g_adv.item(), g_pixel.item())
                )
                torch.save(generator.state_dict(), checkpoint_pth.format()+'.gen')
                torch.save(discriminator.state_dict(), checkpoint_pth.format()+'.disc')
                

def train_similar_images(epochs, trainloader, batch_size, sample_interval, 
            model, criterion, optimizer, checkpoint_pth):
    for epoch in tqdm(range(0, epochs)):
        for i, batch in enumerate(trainloader):
            
            model.train()
            
            full_imgs = batch['full_images'].cuda()
            similar_images = batch['similar_images'].cuda()
            
            mask = batch['masks']
            masked_imgs = full_imgs.detach().clone()
            
            masked_imgs[mask] = 1.0
            
            # Model takes a combination of similar images and masked
            model_input = torch.cat([masked_imgs, similar_images], dim=1)
            
            optimizer.zero_grad()

            # Generate a batch of images
            gen_imgs = model(model_input)

            gen_rois = gen_imgs[mask]
            full_rois = full_imgs[mask]

            recon_loss_full = criterion(gen_imgs, full_imgs)
            recon_loss_roi = criterion(gen_rois, full_rois)
            recon_loss = 0.1*recon_loss_full + 0.9*recon_loss_roi

            recon_loss.backward()
            optimizer.step()

            batches_done = epoch * len(trainloader) + i
            if batches_done % sample_interval == 0:
                print(str(datetime.now()), "[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %f]" % (epoch, epochs, i, len(trainloader), recon_loss.item()))
                r_losses.append(recon_loss.item())
                times.append(time.time())
                show_checkpoint_repros(model)
                torch.save(model.state_dict(), checkpoint_pth.format())

        show_checkpoint_repros_similar_images(model)
        torch.save(model.state_dict(), checkpoint_pth.format())



def train(epochs, trainloader, batch_size, sample_interval, 
            model, criterion, optimizer, checkpoint_pth):
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        for i, batch in tqdm(enumerate(trainloader), total=(len(dataset_train)//batch_size)):
            model.train()
            full_imgs = batch['full_images'].cuda()
            mask = batch['masks']
            masked_imgs = full_imgs.detach().clone()
            
            masked_imgs[mask] = 1.0

            optimizer.zero_grad()

            # Generate a batch of images
            gen_imgs = model(masked_imgs, batch['captions'])
            recon_loss = criterion(gen_imgs, full_imgs)
            epoch_loss+=recon_loss.item()
            
            # gen_rois = gen_imgs[mask]
            # full_rois = full_imgs[mask]

            recon_loss.backward()
            optimizer.step()

            batches_done = epoch * len(trainloader) + i
            if batches_done % sample_interval == 0:
                print(str(datetime.now()), "[Epoch %d/%d] [Batch %d/%d] [Recon Loss: %f]" % (epoch, epochs, i, len(trainloader), recon_loss.item()))
                #save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                r_losses.append(recon_loss.item())
                times.append(time.time())
                show_checkpoint_repros(model)
                pass
            pass
        print(f"Epoch Loss: {epoch_loss/len(dataset_train)}")
        torch.save(model.state_dict(), checkpoint_pth.format())


if __name__ == "__main__":
    # ----------
    #  Set Parameters Based on Model
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--model", type=float, default="ConvAE_cos_channel_attn", help="the model")
    parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--dataset_path", type=str, default='../CUB_200_2011/', help="path to the dataset folder")
    parser.add_argument("--checkpoint_pth", type=str, default='./checkpoint.pth', help="checkpoint file name")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam beta1")
    parser.add_argument("--b2", type=float, default=0.999, help="adam beta2")
    parser.add_argument("--loss_fn", type=float, default="MSELoss", help="the loss function")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
    parser.add_argument("--similar_images", type=bool, default=False, help="flag for related caption approach")
    parser.add_argument("--gan", type=bool, default=False, help="flag for gan approach")
    parser.add_argument("--checkpoint_pth", type=str, default='./checkpoint.pth', help="checkpoint file name")
    parser.add_argument("--embeddings", type=str, default="../corpus_embeddings.pickle", help="path to pickled caption embeddings")
    parser.add_argument("--neighbours", type=str, default="../nearest_neighbors.pickle", help="path to pickled nearest neighbours")

    args = parser.parse_args()

    if (args.similar_images or args.gan):
        dataset_train = CUBDataset_With_Related_Captions(args.dataset_path, 
                                                        normalize=False, 
                                                        embeddings_pickle=args.embeddings, 
                                                        neighbors_pickle=args.neighbours)
        pass
    else:
        dataset_train = CUBDataset(args.dataset_path)
        pass

    trainloader = DataLoader(dataset_train,
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers)

    # Loss function
    if args.loss_fn == 'PSNRLoss':
        criterion = PSNRLoss
        pass
    else:
        criterion = getattr(nn, args.loss_fn)()

    # Initialize generator and discriminator
    model = getattr(models, args.model)

    model.cuda()

    r_losses = []
    times = []

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    if (args.similar_images):
        train_similar_images(args.n_epochs, trainloader, args.batch_size, args.sample_interval, 
            model, criterion, optimizer, args.checkpoint_pth)
        pass
    elif (args.gan):
        # Initialize generator and discriminator
        generator = models.Generator(channels=12)
        discriminator = models.Discriminator(channels=3)

        generator.cuda()
        discriminator.cuda()

        generator.apply(models.weights_init_normal)
        discriminator.apply(models.weights_init_normal)

        train_gan(args.n_epochs, trainloader, args.batch_size, args.sample_interval, 
            generator, discriminator, args.checkpoint_pth)
        pass
    else:
        train(args.n_epochs, trainloader, args.batch_size, args.sample_interval, 
                model, criterion, optimizer, args.checkpoint_pth)

