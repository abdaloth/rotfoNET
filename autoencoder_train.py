import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils.CUBDataset import CUBDataset, image_dim
import autoencoder_models
from torch.utils.data import DataLoader

from tqdm import tqdm
import time
from datetime import datetime
from argparse import ArgumentParser
from PIL import Image
import matplotlib.pyplot as plt

def PSNRLoss(original, inpainted): 
    eps = 1e-11
    mse = torch.mean((original - inpainted)**2)
    psnr = 20 * torch.log10(1/torch.sqrt(mse+eps)) 
    return -psnr


def show_checkpoint_repros(model):
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


# ----------
#  Training
# ----------
  
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
    args = parser.parse_args()

    dataset_train = CUBDataset(args.dataset_path)

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
    model = getattr(autoencoder_models, args.model)

    model.cuda()

    r_losses = []
    times = []

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    train(args.n_epochs, trainloader, args.batch_size, args.sample_interval, 
            model, criterion, optimizer, args.checkpoint_pth)