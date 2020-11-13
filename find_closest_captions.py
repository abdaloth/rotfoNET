from glob import glob
from sentence_transformers.util import pytorch_cos_sim
import torch
from tqdm.notebook import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
for p in embedder.parameters():
    p.requires_grad = False
    pass

text_files = glob("../data/birds/CUB_200_2011/text/*/*.txt")
image_files = glob("../data/birds/CUB_200_2011/images/*/*.jpg")


corpus = []

for file in tqdm(text_files):
    with open(file, "r") as f:
        for line in f.readlines():
            corpus.append(line)


corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True)


with open("./corpus_embeddings.pickle", "wb") as f:
    pickle.dump(corpus_embeddings, f)



with open("./corpus_embeddings.pickle", "rb") as f:
    ce = pickle.load(f)


near_img_idx_dict = dict()

for i in tqdm(range(0,len(corpus))):
    c_emb = embedder.encode([corpus[i]], convert_to_tensor=True)
    closest_3 = torch.topk(pytorch_cos_sim(c_emb, ce), 3)[1][0].cpu().numpy()
    near_img_idx_dict[i] = closest_3


parts = []
for i in tqdm(range(0, len(ce) // 100 + 1)):
    if i != len(ce) // 100:
        parts.append(torch.topk(pytorch_cos_sim(ce[i*100:(i+1)*100], ce), k=3, dim=1)[1])
    else:
        parts.append(torch.topk(pytorch_cos_sim(ce[i*100:], ce), k=3, dim=1)[1])

parts_idxs = [p // 10 for p in parts]

parts_idxs_cpu = [p.cpu() for p in parts_idxs]

all_parts_tensor = torch.cat(parts_idxs_cpu, dim=0)

with open("./nearest_neighbors.pickle", "wb") as f:
    pickle.dump(all_parts_tensor, f)