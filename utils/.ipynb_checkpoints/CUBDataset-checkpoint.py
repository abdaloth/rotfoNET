import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
from  torch.utils.data import Dataset

image_dim = 224

class CUBDataset(Dataset):
    def __init__(self, datadir):
        """
        Dataset of obfuscated coco images, with captions.
        
        annotations: load from pickle, akshay's processed annotations
        datadir: Preprocessed data. Contains /originals and /masked
        tranforms: function to be run on each sample
        """
        
        self.datadir = datadir
        self.image_filenames = sorted(glob(datadir + "images/*/*.jpg"))
        self.text_filenames = sorted(glob(datadir + "text/*/*.txt"))
        self.transform = transforms.Compose([
            transforms.Resize((image_dim,image_dim)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])
        
        self.captions = dict()
        
        for t in self.text_filenames:
            with open(t) as caption_file:
                self.captions[t.split("/")[-1].split(".")[0]] = caption_file.readlines()
        
        # Since every 5 samples is the same image, we have a one image cache.
        # TODO this may get fucky with shuffle? we can find out later.
        self.last_image = None
        self.last_index = None
        
        x1 = 112 - 43
        y1 = 112 - 43
        x2 = x1 + 86
        y2 = y1 + 86
        
        self.mask = torch.zeros(3, image_dim,image_dim,dtype=torch.bool)
        self.mask[:,y1:y2,x1:x2] = True
        
    def __len__(self):
        return len(self.image_filenames) * 10
    
    def __getitem__(self, idx):
        # Load image or retrieve from cache
        image_filename = self.image_filenames[idx // 10]
        image_id = image_filename.split("/")[-1].split(".")[0]
    
        if self.last_index is not None and idx // 10 == self.last_index // 10:
            full_image = self.last_image
        else:
            full_image = Image.open(image_filename)
            self.last_image = full_image

        self.last_index = idx
        full_image = full_image.convert("RGB") # The occasional 1 channel grayscale image is in there.

        caption = self.captions[image_id][idx % 10]

        sample = {
            'captions': caption,
            'full_images': full_image, # Automatically stacked by the loader
            'masks': self.mask # Automatically stacked by the loader
         }

        if self.transform:
            sample['full_images'] = self.transform(sample['full_images'])
            
        return sample