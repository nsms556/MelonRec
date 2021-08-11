import argparse
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

from utils.arena_util import load_json
from utils.static import is_cuda


device = 'cuda' if is_cuda else 'cpu'

def pcc(_x, _y):
    vx = _x - torch.mean(_x)
    vy = _y - torch.mean(_y, axis=1).reshape(-1, 1)
    return torch.sum((vx * vy), axis=1) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum((vy ** 2), axis=1)))

def euclidean(_x, _y):
    return torch.sqrt(torch.sum((_y - _x) ** 2, axis=1))

def calculate_score(train, question, embedding, score_type) :
    all_train_ids = [plylst['id'] for plylst in train]
    all_val_ids = [plylst['id'] for plylst in question]

    train_ids = []
    train_embs = []
    val_ids = []
    val_embs = []

    for plylst_id, emb in tqdm(embedding.items()):
        if plylst_id in all_train_ids:
            train_ids.append(plylst_id)
            train_embs.append(emb)
        elif plylst_id in all_val_ids:
            val_ids.append(plylst_id)
            val_embs.append(emb)

    cos = nn.CosineSimilarity(dim=1)
    train_tensor = torch.tensor(train_embs).to(device)
    val_tensor = torch.tensor(val_embs).to(device)

    scores = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.float64)
    sorted_idx = torch.zeros([val_tensor.shape[0], train_tensor.shape[0]], dtype=torch.int32)

    for idx, val_vector in enumerate(tqdm(val_tensor)):
        if score_type == 'pcc':
            output = pcc(val_vector.reshape(1, -1), train_tensor)
        elif score_type == 'cos':
            output = cos(val_vector.reshape(1, -1), train_tensor)
        elif score_type == 'euclidean':
            output = euclidean(val_vector.reshape(1, -1), train_tensor)
        index_sorted = torch.argsort(output, descending=True)
        scores[idx] = output
        sorted_idx[idx] = index_sorted

    results = defaultdict(list)
    for i, val_id in enumerate(tqdm(val_ids)):
        for j, train_idx in enumerate(sorted_idx[i][:1000]):
            results[val_id].append((train_ids[train_idx], scores[i][train_idx].item()))

    return results

def save_autoencoder_score(train, question, autoencoder_emb, score_type, include_genre, submit_type) :
    score = calculate_score(train, question, autoencoder_emb, score_type)

    if include_genre:
        if submit_type == 'val':
            np.save(f'scores/val_scores_bias_{score_type}_gnr', score)
        elif submit_type == 'test':
            np.save(f'scores/test_scores_bias_{score_type}_gnr', score)
        else:
            np.save(f'scores/local_val_scores_bias_{score_type}_gnr', score)
    else:
        if submit_type == 'val':
            np.save(f'scores/val_scores_bias_{score_type}', score)
        elif submit_type == 'test':
            np.save(f'scores/test_scores_bias_{score_type}', score)
        else:
            np.save(f'scores/local_val_scores_bias_{score_type}', score)

def save_word2vec_score(train, question, word2vec_emb, score_type, submit_type) :
    score = calculate_score(train, question, word2vec_emb, score_type)

    if submit_type == 'local_val':
        np.save(f'scores/local_val_scores_title_{score_type}_24000', score)
    elif submit_type == 'val':
        np.save(f'scores/val_scores_title_{score_type}_24000', score)
    elif submit_type == 'test':
        np.save(f'scores/test_scores_title_{score_type}_24000', score)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, help="local_val: 0, val: 1, test: 2", default=2)

    args = parser.parse_args()
    print(args)

    val_file_path = None
    if args.mode == 0 :
        default_file_path = 'arena_data'
        train_file_path = '{}/orig/train.json'.format(default_file_path)
        question_file_path = '{}/questions/val.json'.format(default_file_path)
        submit_type = 'local_val'
    elif args.mode == 1 :
        default_file_path = 'res'
        train_file_path = '{}/train.json'.format(default_file_path)
        question_file_path = '{}/val.json'.format(default_file_path)
        submit_type = 'val'
    elif args.mode == 2 :
        default_file_path = 'res'
        train_file_path = '{}/train.json'.format(default_file_path)
        val_file_path = '{}/val.json'.format(default_file_path)
        question_file_path = '{}/test.json'.format(default_file_path)
        submit_type = 'test'
    else :
        print('Wrong Mode input')
        exit()

    train = load_json(train_file_path)
    if val_file_path != None :
        train = train + load_json(val_file_path)
    question = load_json(question_file_path)

    autoencoder_emb = np.load('{}/plylst_emb.npy'.format(default_file_path), allow_pickle=True).item()
    autoencoder_emb_gnr = np.load('{}/plylst_emb_gnr.npy'.format(default_file_path), allow_pickle=True).item()
    word2vec_emb = np.load('{}/plylst_w2v_emb.npy'.format(default_file_path), allow_pickle=True).item()

    save_autoencoder_score(train, question, autoencoder_emb, 'cos', False, submit_type)
    save_autoencoder_score(train, question, autoencoder_emb_gnr, 'cos', True, submit_type)
    save_word2vec_score(train, question, word2vec_emb, 'cos', submit_type)

    print('Calculate Similarity Score Complete')