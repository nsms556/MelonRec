# -*- coding: utf-8 -*-
from collections import Counter

import pandas as pd
import numpy as np

import torch


def tags_encoding(json_data, tag2id_path, id2tag_path):
    tag_lists = pd.DataFrame(json_data)['tags'].tolist()

    tags = []
    for tag_list in tag_lists :
        tags.extend(tag_list)
    tags = sorted(list(set(tags)))
  
    tag_to_id = {}
    id_to_tag = {}
    for idx, tag in enumerate(tags) :
        tag_to_id[tag] = idx
        id_to_tag[idx] = tag

    with open(tag2id_path, 'wb') as f:
        np.save(f, tag_to_id)
        print('{} is created'.format(tag2id_path))
    with open(id2tag_path, 'wb') as f:
        np.save(f, id_to_tag)
        print('{} is created'.format(id2tag_path))
    return True


def binary_songs2ids(_input, output, prep_song2id_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()

    to_song_id = lambda x: [prep_song2id_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    songs_idxes = output.argsort(axis=1)[:, ::-1][:, :100]

    return list(map(to_song_id, songs_idxes))


def binary_tags2ids(_input, output, id2tag_dict, istrain=False):
    if torch.cuda.is_available():
        _input = _input.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
    else:
        _input = _input.detach().numpy()
        output = output.detach().numpy()

    to_dict_id = lambda x: [id2tag_dict[_x] for _x in x]

    if not istrain:
        output -= _input

    tags_idxes = output.argsort(axis=1)[:, ::-1][:, :10]

    return list(map(to_dict_id, tags_idxes))


def song_filter_by_freq(train_data, freq_thread, song2id_path, id2song_path) :
    song_counter = Counter()
    for playlist in train_data :
        song_counter.update(playlist['songs'])

    song_counter = list(song_counter.items())

    selected_songs = []
    for song, freq in song_counter :
        if freq > freq_thread :
            selected_songs.append(song)

    freq_song_to_id = {song : _id for _id, song in enumerate(selected_songs)}
    id_to_freq_song = {v:k for k, v in freq_song_to_id.items()}

    np.save(song2id_path, freq_song_to_id)
    np.save(id2song_path, id_to_freq_song)


def genre_gn_all_preprocessing(genre_gn_all):
    ## 대분류 장르코드
    # 장르코드 뒷자리 두 자리가 00인 코드를 필터링
    gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] == '00']

    ## 상세 장르코드
    # 장르코드 뒷자리 두 자리가 00이 아닌 코드를 필터링
    dtl_gnr_code = genre_gn_all[genre_gn_all['gnr_code'].str[-2:] != '00'].copy()
    dtl_gnr_code.rename(columns={'gnr_code': 'dtl_gnr_code', 'gnr_name': 'dtl_gnr_name'}, inplace=True)

    return gnr_code, dtl_gnr_code


def genre_DicGenerator(gnr_code, dtl_gnr_code, song_meta):
    ## gnr_dic (key: 대분류 장르 / value: 대분류 장르 id)
    gnr_dic = {}
    i = 0
    for gnr in gnr_code['gnr_code']:
        gnr_dic[gnr] = i
        i += 1

    ## dtl_dic (key: 상세 장르 / value: 상세 장르 id)
    dtl_dic = {}
    j = 0
    for dtl in dtl_gnr_code['dtl_gnr_code']:
        dtl_dic[dtl] = j
        j += 1

    ## song_gnr_dic (key: 곡 id / value: 해당 곡의 대분류 장르)
    ## song_dtl_dic (key: 곡 id / value: 해당 곡의 상세 장르)
    song_gnr_dic = {}
    song_dtl_dic = {}

    for s in song_meta:
        song_gnr_dic[s['id']] = s['song_gn_gnr_basket']
        song_dtl_dic[s['id']] = s['song_gn_dtl_gnr_basket']

    return gnr_dic, dtl_dic, song_gnr_dic, song_dtl_dic
