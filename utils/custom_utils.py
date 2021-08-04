import os
from tqdm import tqdm

import torch

from arena_util import write_json
from evaluate import ArenaEvaluator
from data_util import binary_songs2ids, binary_tags2ids


def tmp_file_remove(file_path) :
  if os.path.exists(file_path) :
    os.remove(file_path)

def mid_check(q_dataloader, model, tmp_result_path, answer_file_path, id2song_dict, id2tag_dict, is_cuda, num_songs) :
    evaluator = ArenaEvaluator()
    device = 'cuda' if is_cuda else 'cpu'

    tmp_file_remove(tmp_result_path)

    elements =[]
    for idx, (_id, _data) in tqdm(enumerate(q_dataloader), desc='testing...') :
        with torch.no_grad() :
            _data = _data.to(device)
            output = model(_data)

        songs_input, tags_input = torch.split(_data, num_songs, dim=1)
        songs_output, tags_output = torch.split(output, num_songs, dim=1)

        songs_ids = binary_songs2ids(songs_input, songs_output, id2song_dict)
        tags_ids = binary_tags2ids(tags_input, tags_output, id2tag_dict)

        _id = list(map(int, _id))
        for i in range(len(_id)) :
            element = {'id':_id[i], 'songs':list(songs_ids[i]), 'tags':tags_ids[i]}
            elements.append(element)
    
    write_json(elements, tmp_result_path)
    evaluator.evaluate(answer_file_path, tmp_result_path)
    os.remove(tmp_result_path)