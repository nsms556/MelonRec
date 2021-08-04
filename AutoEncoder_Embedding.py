import os
import sys
import argparse
from collections import Counter
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.arena_util import load_json,  write_json
from MelonDataset import *
from utils.data_util import tags_encoding, song_filter_by_freq
from utils.custom_utils import tmp_file_remove, mid_check
from utils.static import is_cuda

class AutoEncoder(nn.Module):
    def __init__(self, D_in, H, D_out, dropout):
        super(AutoEncoder, self).__init__()
        encoder_layer = nn.Linear(D_in, H, bias=True)
        decoder_layer = nn.Linear(H, D_out, bias=True)

        torch.nn.init.xavier_uniform_(encoder_layer.weight)
        torch.nn.init.xavier_uniform_(decoder_layer.weight)

        self.encoder = nn.Sequential(
                        nn.Dropout(dropout),
                        encoder_layer,
                        nn.BatchNorm1d(H),
                        nn.LeakyReLU())
        self.decoder = nn.Sequential(
                        decoder_layer,
                        nn.Sigmoid())

    def forward(self, x):
        out_encoder = self.encoder(x)
        out_decoder = self.decoder(out_encoder)
        return out_decoder
        
class AutoEncoderHandler :
    def __init__(self, model_path:str) -> None:
        self.model = self.set_model(model_path)
        self.device = 'cuda' if is_cuda else 'cpu'

    def __init__(self, args) -> None:
        self.H = args.dimension
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dropout = args.dropout
        self.num_workers = args.num_workers
        self.mode = args.mode

        self.device = 'cuda' if is_cuda else 'cpu'
        self.model = None

    def create_autoencoder(self, D_in, D_out) :
        self.model = AutoEncoder(D_in, self.H, D_out, dropout=self.dropout).to(self.device)

        return self.model

    def set_model(self, model_path) :
        model = torch.load(model_path)
        return model

    def train_autoencoder(self, train_dataset, autoencoder_model_path, id2song_file_path, id2tag_file_path, question_dataset, answer_file_path) :
        id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
        id2song_dict = dict(np.load(id2song_file_path, allow_pickle=True).item())

        num_songs = train_dataset.num_songs
        num_tags = train_dataset.num_tags

        D_in = D_out = num_songs + num_tags

        q_dataloader = None
        check_every = 5
        tmp_result_path = 'results/tmp_results.json'

        if question_dataset is not None :
            q_dataloader = DataLoader(question_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

        dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

        model = self.create_autoencoder(D_in, D_out)
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        try :
            model = torch.load(autoencoder_model_path)
            print('Use exist AutoEncoder Model')
        except :
            print('AutoEncoder Model not found')
            pass

        temp_fn = 'arena_data/answers/temp.json'
        tmp_file_remove(temp_fn)

        for epoch in range(args.epochs) :
            print('epoch : {}'.format(epoch))

            running_loss = 0.0
            for idx, (_id, _data) in tqdm(enumerate(dataloader), desc='training...') :
                _data = _data.to(self.device)

                optimizer.zero_grad()
                output = model(_data)

                loss = loss_func(output, _data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print('loss: {:.4f} \n'.format(running_loss))

            torch.save(model, autoencoder_model_path)

            if args.mode == 0 & epoch % check_every == 0 :
                mid_check(q_dataloader, model, tmp_result_path, answer_file_path, id2song_dict, id2tag_dict)

    def autoencoder_plylsts_embeddings(self, _model_file_path, _submit_type, genre=False, use_exist=True):
        if _submit_type == 'val':
            default_file_path = 'res'
            question_file_path = 'res/val.json'
            train_file_path = 'res/train.json'
            val_file_path = 'res/val.json'
            train_dataset = load_json(train_file_path)
        elif _submit_type == 'test':
            default_file_path = 'res'
            question_file_path = 'res/test.json'
            train_file_path = 'res/train.json'
            val_file_path = 'res/val.json'
            train_dataset = load_json(train_file_path) + load_json(val_file_path)
        elif _submit_type == 'local_val':
            default_file_path = 'arena_data'
            train_file_path = f'{default_file_path}/orig/train.json'
            question_file_path = f'{default_file_path}/questions/val.json'
            train_dataset = load_json(train_file_path)

        question_dataset = load_json(question_file_path)

        tag2id_file_path = f'{default_file_path}/tag2id_{_submit_type}.npy'
        prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr2_{_submit_type}.npy'

        if genre:
            train_dataset = SongTagGenreDataset(train_dataset, tag2id_file_path, prep_song2id_file_path)
            question_dataset = SongTagGenreDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)
        else:
            train_dataset = SongTagDataset(train_dataset, tag2id_file_path, prep_song2id_file_path)
            question_dataset = SongTagDataset(question_dataset, tag2id_file_path, prep_song2id_file_path)

        plylst_embed_weight = []
        plylst_embed_bias = []

        model_file_path = _model_file_path

        if use_exist :
            model = self.set_model(model_file_path)
        else :
            model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == 'encoder.1.weight':
                    plylst_embed_weight = param.data
                elif name == 'encoder.1.bias':
                    plylst_embed_bias = param.data

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=4)
        question_loader = DataLoader(question_dataset, shuffle=True, batch_size=256, num_workers=4)

        plylst_emb_with_bias = dict()

        if genre:
            for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(train_loader, desc='get train vectors...')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                    output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]

            for idx, (_id, _data, _dnr, _dtl_dnr) in enumerate(tqdm(question_loader, desc='get question vectors...')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()
                    output_with_bias = np.concatenate([output_with_bias, _dnr, _dtl_dnr], axis=1)

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]
        else:
            for idx, (_id, _data) in enumerate(tqdm(train_loader, desc='get train vectors...')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]

            for idx, (_id, _data) in enumerate(tqdm(question_loader, desc='get question vectors...')):
                with torch.no_grad():
                    _data = _data.to(self.device)
                    output_with_bias = (torch.matmul(_data, plylst_embed_weight.T) + plylst_embed_bias).tolist()

                    _id = list(map(int, _id))
                    for i in range(len(_id)):
                        plylst_emb_with_bias[_id[i]] = output_with_bias[i]

        return plylst_emb_with_bias

def get_file_paths(args) :
    answer_file_path = None

    if args.mode == 0: 
        default_file_path = 'arena_data'
        model_postfix = 'local_val'
        train_file_path = f'{default_file_path}/orig/train.json'
        question_file_path = f'{default_file_path}/questions/val.json'
        answer_file_path = f'{default_file_path}/answers/val.json'
        
    elif args.mode == 1:
        default_file_path = 'res'
        model_postfix = 'val'
        train_file_path = f'{default_file_path}/train.json'
        val_file_path = f'{default_file_path}/val.json'
        
    elif args.mode == 2:
        default_file_path = 'res'
        model_postfix = 'test'
        train_file_path = f'{default_file_path}/train.json'
        val_file_path = f'{default_file_path}/val.json'
        test_file_path = f'{default_file_path}/test.json'
        
    else:
        print('mode error! local_val: 0, val: 1, test: 2')
        sys.exit(1)

    tag2id_file_path = f'{default_file_path}/tag2id_{model_postfix}.npy'
    id2tag_file_path = f'{default_file_path}/id2tag_{model_postfix}.npy'
    song2id_file_path = f'{default_file_path}/freq_song2id_thr{args.freq_thr}_{model_postfix}.npy'
    id2song_file_path = f'{default_file_path}/id2freq_song_thr{args.freq_thr}_{model_postfix}.npy'

    autoencoder_model_path = 'model/autoencoder_{}_{}_{}_{}_{}_{}.pkl'. \
        format(args.H, args.batch_size, args.learning_rate, args.dropout, args.freq_thr, model_postfix)

    return train_file_path, val_file_path, test_file_path, question_file_path, answer_file_path, \
        tag2id_file_path, id2tag_file_path, song2id_file_path, id2song_file_path, autoencoder_model_path


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=450)
    parser.add_argument('-epochs', type=int, help="total epochs", default=41)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=20)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-mode', type=int, help="local_val: 0, val: 1, test: 2", default=2)

    args = parser.parse_args()
    print(args)

    mode = args.mode

    train_file_path, val_file_path, test_file_path, question_file_path, answer_file_path, \
        tag2id_file_path, id2tag_file_path, song2id_file_path, id2song_file_path, autoencoder_model_path = get_file_paths(args)

    handler = AutoEncoderHandler()

    if mode == 0 :
        default_file_path = 'arena_data'
        model_postfix = 'local_val'
        train_data = load_json(train_file_path)
        question_data = load_json(question_file_path)
    elif mode == 1 :
        default_file_path = 'res'
        model_postfix = 'val'
        train_data = load_json(train_file_path) + load_json(val_file_path)
    elif mode == 2 :
        default_file_path = 'res'
        model_postfix = 'test'
        train_data = load_json(train_file_path) + load_json(val_file_path) + load_json(test_file_path)

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_encoding(train_data, tag2id_file_path, id2tag_file_path)

    if not (os.path.exists(song2id_file_path) & os.path.exists(id2song_file_path)):
        song_filter_by_freq(train_data, args.freq_thr, song2id_file_path, id2song_file_path)

    train_dataset = SongTagDataset(train_data, tag2id_file_path, song2id_file_path)

    if question_data is not None :
        question_dataset = SongTagDataset(question_data, tag2id_file_path, song2id_file_path)

    handler.train_autoencoder(train_dataset, autoencoder_model_path, id2song_file_path, id2tag_file_path, question_dataset, answer_file_path)

    plylst_emb = handler.autoencoder_plylsts_embeddings(autoencoder_model_path, model_postfix, False, False)
    plylst_emb_gnr = handler.autoencoder_plylsts_embeddings(autoencoder_model_path, model_postfix, True, False)

    np.save('{}/plylst_emb.npy'.format(default_file_path), plylst_emb)
    np.save('{}/plylst_emb_gnr.npy'.format(default_file_path), plylst_emb_gnr)
