from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import kss
import argparse
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').cuda()

model.eval()

texts = pd.read_csv("data/sentence_preprocessed.csv")['text'].values

prompt = input('prompt : ')
prompt_embeddings = model.encode(prompt)


sentences = []
sim_list = []

for i in tqdm(texts):
    local_text = kss.split_sentences(i)
    local_embeddings = model.encode(local_text)  # (sentence_num, 768)

    for index in range(len(local_text)):
        sentence = ''
        try:
            assert index > 0
            sentence += (local_text[index-1] + ' ')
        except:
            pass

        sentence += (local_text[index] + ' ')

        try:
            sentence += (local_text[index+1] + ' ')
        except:
            pass

        sentences.append(sentence)

    for emb in local_embeddings:  # 1st embedding , 2nd embeddings, ...
        sim_list.append(cosine_similarity(emb, prompt_embeddings))

data = pd.DataFrame({'sentence' : [None], 'similarity' : [None]})

data = data[:0]

data['sentence'] = sentences
data['similarity'] = sim_list

data.to_csv('data/sentence_similarity.csv', encoding='utf8')
