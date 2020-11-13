import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer

# GAN code reference: https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/context_encoder
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, channels=12): # 3 for original image, plus 9 for 3 related images
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *downsample(channels, 64, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            nn.Conv2d(64, 3, 3, 1, 1), # channels output used to be same as channels  input
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(64, 2, False), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)



class ConvAE_cos_channel_attn(nn.Module):
    """
    Convolutional AutoEncoder with text-image channel attention
    """
    def __init__(self):
        super(ConvAE_cos_channel_attn,self).__init__()
        self.embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        for p in self.embedder.parameters():
            p.requires_grad = False
            pass
        
        self.e1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.e2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
            
        )
        
        self.caption_fc = nn.Linear(768,24*24)
        
        self.d1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.d2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.d3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d5=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x, caption):
        x=self.e1(x)
        x=self.e2(x)
        x=self.e3(x)
        x=self.e4(x)
        x=self.e5(x)
        encoded_caption = self.embedder.encode(caption, convert_to_tensor=True).cuda()
        encoded_caption = self.caption_fc(encoded_caption)
        
        cos_sim = F.cosine_similarity(x.view(x.size(0), x.size(1), 24*24), encoded_caption.view(x.size(0), 1, 24*24), -1).clamp(min=0)
        cos_sim = F.softmax(cos_sim, dim=1)
        
        x = cos_sim.view(x.size(0), -1, 1, 1) * x
        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.d4(x)
        x=self.d5(x)
        return x


class ConvAE(nn.Module):

    """
    Convolutional AutoEncoder with concatenated embeddings
    """
    def __init__(self):
        super(ConvAE,self).__init__()
        self.embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        for p in self.embedder.parameters():
            p.requires_grad = False
            pass
        
        self.e1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.e2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
            
        )
        
        self.caption_fc = nn.Linear(768,24*24)
        
        self.d1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=513,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.d2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.d3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d5=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x, caption):
        x=self.e1(x)
        x=self.e2(x)
        x=self.e3(x)
        x=self.e4(x)
        x=self.e5(x)
        encoded_caption = self.embedder.encode(caption, convert_to_tensor=True).cuda()
        encoded_caption = self.caption_fc(encoded_caption)
        x = torch.cat((encoded_caption.view(-1, 1, 24, 24), x), 1)
        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.d4(x)
        x=self.d5(x)
        x = (x + 1) / 2
        return x #output of generator


class ConvAE_custom_embeddings(nn.Module):
    """
    Convolutional AE that learns embeddings 
    """
    #generator model
    def __init__(self, vocab_dict):
        super(ConvAE_custom_embeddings,self).__init__()
        self.embedder = nn.Embedding(num_embeddings=len(vocab_dict), embedding_dim=50, padding_idx=0)
        self.gru = nn.GRU(50, 100, batch_first=True, bidirectional=True)
        self.caption_fc = nn.Linear(200, 24*24)
        
        self.e1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.e2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.e5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
            
        )
        
        self.d1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=513,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.d2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.d3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.d5=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x, caption):
        x=self.e1(x)
        x=self.e2(x)
        x=self.e3(x)
        x=self.e4(x)
        x=self.e5(x)
        encoded_caption = self.embedder(caption)
        h0 = torch.zeros(2, x.shape[0], 100).cuda()
        encoded_caption, h0 = self.gru(encoded_caption, h0)
        encoded_caption = self.caption_fc(encoded_caption)
        encoded_caption = torch.mean(encoded_caption, 1)
        x = torch.cat((encoded_caption.view(-1, 1, 24, 24), x), 1)
        x=self.d1(x)
        x=self.d2(x)
        x=self.d3(x)
        x=self.d4(x)
        x=self.d5(x)
        x = (x + 1) / 2
        return x #output of generator


class Vanilla_AE(nn.Module):
    """
    Convolutional AE implementation without embedding
    """
    def __init__(self):
        super(Vanilla_AE, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5,5))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), return_indices=True)
        self.unconv1 = nn.ConvTranspose2d(6,3,kernel_size=(5,5))
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=(2,2))
        self.unmaxunpool2 = nn.MaxUnpool2d(kernel_size=(2,2))
        
        self.encoder1 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(6, 12,kernel_size=(5,5)),
        )
        
        self.encoder2 = nn.Sequential(
            nn.Tanh(),
            nn.Conv2d(12, 16, kernel_size=(5,5)),
            nn.Tanh()
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(16, 12, kernel_size=(5,5)),
            nn.Tanh()
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(12,6,kernel_size=(5,5)),
            nn.Tanh(),
        )
        

    def forward(self, x):
        x = self.conv1(x)
        x,indices1 = self.maxpool1(x)
        x = self.encoder1(x)
        x,indices2 = self.maxpool2(x)
        x = self.encoder2(x)
        
        x = self.decoder2(x)
        x = self.unmaxunpool2(x, indices2)
        x = self.decoder1(x)
        x = self.maxunpool1(x,indices1)
        x = self.unconv1(x)
        x = nn.Tanh()(x)
        return x



class ConvAE_Similar_Images(nn.Module):
    """
    Convolutional AE using captions to find similar images
    """
    
    #generator model
    def __init__(self):
        super(ConvAE_Similar_Images,self).__init__()
        
        self.t1=nn.Sequential(
            # 12 channels - 3 for original, 9 for 3 additional images.
            nn.Conv2d(in_channels=12,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.LeakyReLU(0.2,inplace=True)
        )
        
        self.t2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.t3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.t4=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )
        self.t5=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True)
            
        )
        
        self.t7=nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.t8=nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(5,5),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.t9=nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.t10=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.t11=nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        x=self.t1(x)
        x=self.t2(x)
        x=self.t3(x)
        x=self.t4(x)
        x=self.t5(x)
        x=self.t7(x)
        x=self.t8(x)
        x=self.t9(x)
        x=self.t10(x)
        x=self.t11(x)
        x = (x + 1) / 2
        return x #output of generator        