import torch
import torch.nn as nn
import csv
import numpy as np
import json
import os
import re

np.random.seed(42)
embedding_dim = 384

def load_templates(file_path):
    template_dict = {}
    tid_dict = {}
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader) 
        tid = 4 # start from 2. 0 for <AGG>, 1 for <EOS>, 2 for <PAD>
        tid_dict['<AGG>'] = 0
        tid_dict['<EOS>'] = 1
        tid_dict['<PAD>'] = 2
        tid_dict['<MASK>'] = 3
        for row in csv_reader:
            print(row)
            template_dict[row[0]] = row[1]
            tid_dict[row[0]] = tid
            tid = tid + 1
    return template_dict, tid_dict

base_path = "./dataset/HDFS/"
template_filename = 'HDFS.log_templates.csv'
id_temp_dict, tid_id_dict = load_templates(os.path.join(base_path, template_filename))

# Save the mapping to a JSON file
output_base_path = "./dataset/HDFS/"

os.makedirs(output_base_path, exist_ok=True)
with open(os.path.join(output_base_path, 'template_id_mapping.json'), 'w') as f:
    json.dump(tid_id_dict, f)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_template(tmp, emb_dim):
    if tmp == 'random':
        return np.random.rand(1, emb_dim)

    tmp = tmp.lower()
    tmp = re.sub(r'[^a-zA-Z\s,]', ' ', tmp) 

    # print(tmp)
    output_emb = model.encode(tmp)
    # print(output_emb.shape)
    return output_emb


    # print(last_hidden_states.shape)
    # return np.random.rand(1, emb_dim)

def emb_generation(id_temp_dict, tid_id_dict):
    embedding_dim = 384
    emb_lookup = np.zeros((len(id_temp_dict)+4, embedding_dim), np.float32)
    for t_id, temp in id_temp_dict.items():
        emb_lookup[tid_id_dict[t_id]] = embed_template(temp, embedding_dim)
        # return
    emb_lookup[tid_id_dict['<AGG>']] = embed_template('random', embedding_dim)
    emb_lookup[tid_id_dict['<EOS>']] = embed_template('random', embedding_dim)
    emb_lookup[tid_id_dict['<PAD>']] = embed_template('random', embedding_dim)
    emb_lookup[tid_id_dict['<MASK>']] = embed_template('random', embedding_dim)
    return emb_lookup

emb_lookup = emb_generation(id_temp_dict, tid_id_dict)

embedding_layer = nn.Embedding(len(id_temp_dict)+4, embedding_dim)


embedding_layer.weight = nn.Parameter(torch.tensor(emb_lookup, dtype=torch.float32), requires_grad=False)

torch.save(embedding_layer.state_dict(), os.path.join(output_base_path, 'sentence_embedding_params.pth'))
