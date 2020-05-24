import numpy as np
from torch.utils.data import Dataset, DataLoader
from arena_util import load_json
import torch


class SongTagDataset(Dataset):
    def __init__(self, file_path=None):
        if file_path:
            self.train = load_json(file_path)
        else:
            self.train = load_json('arena_data/orig/train.json')
        self.tag_to_id = dict(np.load('arena_data/orig/tag_to_id.npy', allow_pickle=True).item())
        self.freq_song_to_id = dict(np.load('arena_data/orig/freq_song_to_id.npy', allow_pickle=True).item())
        self.num_songs = len(self.freq_song_to_id)
        self.num_tags = 29160

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        song_vector = self.song_ids2vec(self.train[idx]['songs'])
        tag_vector = self.tag_ids2vec(self.train[idx]['tags'])
        _input = torch.from_numpy(np.concatenate([song_vector, tag_vector]))
        return _input

    def song_ids2vec(self, songs):
        songs = [self.freq_song_to_id[song] for song in songs if song in self.freq_song_to_id.keys()]
        songs = np.asarray(songs)
        bin_vec = np.zeros(self.num_songs)
        if len(songs) > 0:
            bin_vec[songs] = 1
        return np.array(bin_vec, dtype=np.float32)
            
    def tag_ids2vec(self, tags):
        tags = [self.tag_to_id[tag] for tag in tags]
        tags = np.asarray(tags)
        bin_vec = np.zeros(self.num_tags)
        bin_vec[tags] = 1
        return np.array(bin_vec, dtype=np.float32)
