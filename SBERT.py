# !pip install sentence_transformers
# !pip install kss

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from tqdm import tqdm
import kss
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS').cuda()

model.eval()

data = pd.read_csv('data/train_fold.csv', index_col=0)

texts = pd.read_csv("data/train_fold.csv")['text'].values

prompt = input('prompt : ')
prompt_embeddings = model.encode(prompt)

mean_cosine = []
max_cosine = []
for i in tqdm(texts):
    local_text = kss.split_sentences(i)
    local_embeddings = model.encode(local_text)  # (sentence_num, 768)

    sim_list = []
    for emb in local_embeddings:  # 1st embedding , 2nd embeddings, ...
        sim_list.append(cosine_similarity(emb, prompt_embeddings))

    local_mean = np.mean(sim_list)
    local_max = np.max(sim_list)

    mean_cosine.append(local_mean)
    max_cosine.append(local_max)


data['mean_similarity'] = mean_cosine
data['max_similarity'] = max_cosine

data.to_csv('data/train_fold_embeddings.csv', encoding='utf8')