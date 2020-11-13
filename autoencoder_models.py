import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer


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