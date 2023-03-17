# -*- coding: UTF-8 -*-
# author    : huanghui
import os
import json
import random
import re

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    res = f"{int(s)}s"
    if m > 0:
        res = f"{int(m)}m:" + res
    if h > 0:
        res = f"{int(h)}h:" + res
    return res


def check_dir(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def write_to_file(data, filename):
    open(filename, 'w', encoding='utf-8').write(data)


def write_to_json(data, filename, by_line=False, default=None):
    if by_line:
        data = [json.dumps(d, ensure_ascii=False, default=default) for d in data]
        write_to_file("\n".join(data), filename)
    else:
        write_to_file(json.dumps(data, ensure_ascii=False, indent=4, default=default), filename)


def load_from_json(filename, by_line=False):
    if by_line:
        data = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    return json.loads(open(filename, 'r', encoding='utf-8').read())


def get_label_map(label_file, use_bio=False, default_none='O'):
    labels = json.load(open(label_file, "r", encoding='utf-8'))
    if use_bio:
        labels = get_bio_labels(labels)
    elif default_none is not None:
        labels = [default_none] + labels

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def get_bio_labels(ori_labels):
    labels = ['O']
    for label in ori_labels:
        labels.extend([f'B-{label}', f'I-{label}'])
    return labels


def get_model_path_list(base_dir):
    model_lists = []
    for sub_dir in os.listdir(base_dir):
        if sub_dir.startswith("checkpoint"):
            step = sub_dir.split('-')[-1]
            if re.match("[0-9]+", step):
                model_lists.append((os.path.join(base_dir, sub_dir), int(step)))

    model_lists = sorted(model_lists, key=lambda x: x[1])
    model_lists = [sub_dir[0] for sub_dir in model_lists]

    return model_lists


def load_embeddings(embedding_file):
    vocab2id = {}
    embedding = []
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            vector = list(map(float, line[1:]))
            word = line[0]
            vocab2id[word] = len(vocab2id)
            embedding.append(vector)
    return vocab2id, np.array(embedding)
