# -*- coding: utf-8 -*-

import os
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from MelonDataset import SongTagDataset, SongTagGenreDataset, SongTagTimeDataset
from Models import AutoEncoder
from data_util import *
from arena_util import write_json
from evaluate import ArenaEvaluator

def train_with_time(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())

    qestion_data_loader = None
    if not submit:
        qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size,
                                         num_workers=num_workers)

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags
    num_year = train_dataset.num_year
    num_month = train_dataset.num_month

    # hyper parameters
    D_in = D_out = num_songs + num_tags + num_year + num_month

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)

    testevery = 5

    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        model = torch.load(model_file_path)
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    temp_fn = 'arena_data/answers/temp.json'
    if os.path.exists(temp_fn):
        os.remove(temp_fn)

    for epoch in range(epochs):
        print()
        print('epoch: ', epoch)
        running_loss = 0.0
        for idx, (_id, _data) in enumerate(tqdm(data_loader, desc='training...')):
            _data = _data.to(device)

            optimizer.zero_grad()
            output = model(_data)
            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)
        if not submit:
            if epoch % testevery == 0:
                elements = []
                for idx, (_id, _data) in enumerate(tqdm(qestion_data_loader, desc='testing...')):
                    with torch.no_grad():
                        _data = _data.to(device)
                        output = model(_data)

                        songs_tags_input, _ = torch.split(_data, num_songs + num_tags, dim=1)
                        songs_input, tags_input = torch.split(songs_tags_input, num_songs, dim=1)
                        songs_and_tags_output, _ = torch.split(output, num_songs + num_tags, dim=1)
                        songs_output, tags_output = torch.split(songs_and_tags_output, num_songs, dim=1)

                        songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
                        tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

                        _id = list(map(int, _id))
                        for i in range(len(_id)):
                            element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                            elements.append(element)

                write_json(elements, temp_fn)
                evaluator.evaluate(answer_file_path, temp_fn)
                os.remove(temp_fn)


def train_with_genre(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())

    qestion_data_loader = None
    if not submit:
        qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size,
                                         num_workers=num_workers)

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags
    num_gnr = train_dataset.num_gnr
    num_dtl_gnr = train_dataset.num_dtl_gnr

    # hyper parameters
    D_in = D_out = num_songs + num_tags + num_gnr + num_dtl_gnr

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)

    testevery = 5

    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        model = torch.load(model_file_path)
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    temp_fn = 'arena_data/answers/temp.json'
    if os.path.exists(temp_fn):
        os.remove(temp_fn)

    for epoch in range(epochs):
        print()
        print('epoch: ', epoch)
        running_loss = 0.0
        for idx, (_id, _data) in enumerate(tqdm(data_loader, desc='training...')):
            _data = _data.to(device)

            optimizer.zero_grad()
            output = model(_data)
            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)
        if not submit:
            if epoch % testevery == 0:
                elements = []
                for idx, (_id, _data) in enumerate(tqdm(qestion_data_loader, desc='testing...')):
                    with torch.no_grad():
                        _data = _data.to(device)
                        output = model(_data)

                        songs_tags_input, _ = torch.split(_data, num_songs + num_tags, dim=1)
                        songs_input, tags_input = torch.split(songs_tags_input, num_songs, dim=1)
                        songs_and_tags_output, _ = torch.split(output, num_songs + num_tags, dim=1)
                        songs_output, tags_output = torch.split(songs_and_tags_output, num_songs, dim=1)

                        songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
                        tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

                        _id = list(map(int, _id))
                        for i in range(len(_id)):
                            element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                            elements.append(element)

                write_json(elements, temp_fn)
                evaluator.evaluate(answer_file_path, temp_fn)
                os.remove(temp_fn)


def train_without_genre(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path, model_file_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id2tag_dict = dict(np.load(id2tag_file_path, allow_pickle=True).item())
    id2prep_song_dict = dict(np.load(id2prep_song_file_path, allow_pickle=True).item())

    qestion_data_loader = None
    if not submit:
        qestion_data_loader = DataLoader(question_dataset, shuffle=True, batch_size=batch_size,
                                         num_workers=num_workers)

    # parameters
    num_songs = train_dataset.num_songs
    num_tags = train_dataset.num_tags

    # hyper parameters
    D_in = D_out = num_songs + num_tags

    evaluator = ArenaEvaluator()
    data_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    model = AutoEncoder(D_in, H, D_out, dropout=dropout).to(device)

    testevery = 5

    parameters = model.parameters()
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    try:
        model = torch.load(model_file_path)
        print("\n--------model restored--------\n")
    except:
        print("\n--------model not restored--------\n")
        pass

    temp_fn = 'arena_data/answers/temp.json'
    if os.path.exists(temp_fn):
        os.remove(temp_fn)

    for epoch in range(epochs):
        print()
        print('epoch: ', epoch)
        running_loss = 0.0
        for idx, (_id, _data) in enumerate(tqdm(data_loader, desc='training...')):
            _data = _data.to(device)

            optimizer.zero_grad()
            output = model(_data)
            loss = loss_func(output, _data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss: %d %d%% %.4f' % (epoch, epoch / epochs * 100, running_loss))

        torch.save(model, model_file_path)
        if not submit:
            if epoch % testevery == 0:
                elements = []
                for idx, (_id, _data) in enumerate(tqdm(qestion_data_loader, desc='testing...')):
                    with torch.no_grad():
                        _data = _data.to(device)
                        output = model(_data)

                        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
                        songs_output, tags_output = torch.split(output, num_songs, dim=1)

                        songs_ids = binary_songs2ids(songs_input, songs_output, id2prep_song_dict)
                        tag_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

                        _id = list(map(int, _id))
                        for i in range(len(_id)):
                            element = {'id': _id[i], 'songs': list(songs_ids[i]), 'tags': tag_ids[i]}
                            elements.append(element)

                write_json(elements, temp_fn)
                evaluator.evaluate(answer_file_path, temp_fn)
                os.remove(temp_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dimension', type=int, help="hidden layer dimension", default=510)
    parser.add_argument('-epochs', type=int, help="total epochs", default=31)
    parser.add_argument('-batch_size', type=int, help="batch size", default=256)
    parser.add_argument('-learning_rate', type=float, help="learning rate", default=0.0005)
    parser.add_argument('-dropout', type=float, help="dropout", default=0.2)
    parser.add_argument('-num_workers', type=int, help="num workers", default=20)
    parser.add_argument('-freq_thr', type=float, help="frequency threshold", default=2)
    parser.add_argument('-input_type', type=float, help="2: org + update time, 1: org + genre, 0: original(song + tag)", default=0)
    parser.add_argument('-submit', type=int, help="arena_data/orig: 0 res: 1", default=0)

    args = parser.parse_args()
    print(args)

    H = args.dimension
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    num_workers = args.num_workers
    freq_thr = args.freq_thr
    input_type = args.input_type
    submit = args.submit

    if submit:
        default_file_path = 'res'
        question_file_path = 'res/val.json'
        model_postfix = '_sub'
        train_file_path = f'{default_file_path}/train_val.json'
    else:
        default_file_path = 't_arena_data/orig'
        question_file_path = 't_arena_data/questions/t_val.json'
        model_postfix = ''
        train_file_path = f'{default_file_path}/t_train.json'

    answer_file_path = 't_arena_data/answers/t_val.json'

    tag2id_file_path = f'{default_file_path}/tag2id.npy'
    id2tag_file_path = f'{default_file_path}/id2tag.npy'

    prep_song2id_file_path = f'{default_file_path}/freq_song2id_thr{freq_thr}.npy'
    id2prep_song_file_path = f'{default_file_path}/id2freq_song_thr{freq_thr}.npy'

    if not (os.path.exists(tag2id_file_path) & os.path.exists(id2tag_file_path)):
        tags_ids_convert(train_file_path, tag2id_file_path, id2tag_file_path)

    if input_type == 2:
        train_dataset = SongTagTimeDataset(train_file_path, tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagTimeDataset(question_file_path, tag2id_file_path, prep_song2id_file_path)
        model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}{}_time_t.pkl'. \
            format(H, batch_size, learning_rate, dropout, freq_thr, model_postfix)
        train_with_time(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path,
                         model_file_path)
    elif input_type == 1:
        train_dataset = SongTagGenreDataset(train_file_path, tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagGenreDataset(question_file_path, tag2id_file_path, prep_song2id_file_path)
        model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}_gnr_t{}.pkl'. \
            format(H, batch_size, learning_rate, dropout, freq_thr, model_postfix)
        train_with_genre(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path, model_file_path)
    else:
        train_dataset = SongTagDataset(train_file_path, tag2id_file_path, prep_song2id_file_path)
        question_dataset = SongTagDataset(question_file_path, tag2id_file_path, prep_song2id_file_path)
        model_file_path = 'model/autoencoder_{}_{}_{}_{}_{}_t{}.pkl'. \
            format(H, batch_size, learning_rate, dropout, freq_thr, model_postfix)
        train_without_genre(train_dataset, id2prep_song_file_path, id2tag_file_path, question_dataset, answer_file_path, model_file_path)

    print('train completed')
