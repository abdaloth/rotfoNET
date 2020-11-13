import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob
from  torch.utils.data import Dataset
from sentence_transformers.util import pytorch_cos_sim
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle

image_dim = 224

class CUBDataset_With_Related_Captions(Dataset):
    def __init__(self, datadir, normalize=True, num_related=3, show_progress_bar=True, embeddings_pickle="./corpus_embeddings.pickle", device=torch.cuda.current_device(), neighbors_pickle="./nearest_neighbors.pickle"):
        """
        This version of CUBDataset also gives similar images from the training set, as defined by the cosine similarity of the embedded caption.
        """
        
        self.datadir = datadir
        self.device = device
        self.num_related = num_related
        self.image_filenames = sorted(glob(datadir + "images/*/*.jpg"))
        self.text_filenames = sorted(glob(datadir + "text/*/*.txt"))
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize((image_dim,image_dim)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_dim,image_dim)),
                transforms.ToTensor()
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

        # Construct sentence transformer
        self.embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device=self.device)
        for p in  self.embedder.parameters(): # Unsure why they come with grad on, but we turn it off
            p.requires_grad = False
            pass

        if embeddings_pickle == None:
            # Build caption corpus
            corpus = []
            for file in self.text_filenames:
                with open(file, "r") as f:
                    for line in f.readlines():
                        corpus.append(line)

            self.corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True, device=self.device) # This line takes ~ 3 minutes to run.
        else:
            with open(embeddings_pickle, "rb") as f:
                self.corpus_embeddings = pickle.load(f)
                
        with open(neighbors_pickle, "rb") as f:
            self.nearest_neighbors_tensor = pickle.load(f)
        
    def getNearestImages_BAD(self, caption):
        """
        Given a caption string, returns a num_related x image_dim x image_dim pytorch FloatTensor
        with num_related best matched images according to their captions.
        """
        
        caption_embedding = self.embedder.encode(caption, convert_to_tensor=True, device=self.device)
        cos_scores = pytorch_cos_sim(caption_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        top_results = torch.topk(cos_scores, k=self.num_related)
        
        indices = top_results[1] // 10

        images = []
        for i in indices:
            im = Image.open(self.image_filenames[i])
            im = im.convert("RGB") # The occasional 1 channel grayscale image is in there.
            if self.transform:
                im = self.transform(im)
            images.append(im)

        return torch.cat(images) # It's just 3 * num_related channels, e.g. 9x224x224
    
    def getNearestImages(self, caption_index):
        nearest_neighbor_indices = self.nearest_neighbors_tensor[caption_index] # Loaded lookup table.

        images = []
        for i in nearest_neighbor_indices:
            im = Image.open(self.image_filenames[i])
            im = im.convert("RGB") # The occasional 1 channel grayscale image is in there.
            if self.transform:
                im = self.transform(im)
            images.append(im)

        return torch.cat(images) # It's just 3 * num_related channels, e.g. 9x224x224

    def getNearestImageIndices_BAD(self, caption):
        """
        Given a caption string, returns a num_related x image_dim x image_dim pytorch FloatTensor
        with num_related best matched images according to their captions.
        """

        caption_embedding = self.embedder.encode(caption, convert_to_tensor=True)
        cos_scores = pytorch_cos_sim(caption_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        top_results = torch.topk(cos_scores, k=self.num_related)
        
        indices = top_results[1] // 10

        return indices

        
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
        
        # Using the caption, load images we believe will be similar based on having similar captions.
        similar_images = self.getNearestImages(idx)
        #similar_images = similar_images.cpu() # This was a weird error - things apparently need to be on CPU when they come out of the loader.
        
        sample = {
            'captions': caption,
            'full_images': full_image, # Automatically stacked by the loader
            'similar_images':similar_images, # Automatically stacked by the loader
            'masks': self.mask # Automatically stacked by the loader
         }

        if self.transform:
            sample['full_images'] = self.transform(sample['full_images'])
            
        return sample