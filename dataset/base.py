# -*- coding: UTF-8 -*-
# author    : huanghui
import hashlib
import os
import logging
import re
from typing import List, Dict, Union
from dataclasses import dataclass
import json
import numpy as np
import copy
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
from torch.utils.data import Dataset
from abc import ABC
import pickle
import gzip

logger = logging.getLogger("main")


@dataclass
class Span:
    start: int
    end: int
    label: int = None


@dataclass
class Event:
    trigger: Span
    arguments: List[Span]


@dataclass
class InputExample:
    doc_id: str = None
    sent_id: str = None
    words: List[str] = None
    events: List[Event] = None
    entities: List[Span] = None
    left: List[str] = None
    right: List[str] = None


@dataclass
class Document:
    doc_key: str = None
    examples: List[InputExample] = None


@dataclass
class InputFeature:
    guid: str = None
    word_ids: List[str] = None
    input_ids: List[int] = None
    attention_mask: List[int] = None
    token_type_ids: List[int] = None
    pieces2word: np.ndarray = None
    labels: List[np.ndarray] = None
    offsets: List = None
    remark: str = None


@dataclass
class Result:
    doc_id: str = None
    sent_id: str = None
    events: List[Event] = None


class BaseDataset(Dataset, ABC):
    cache_dir = "cache"
    input_names = ['input_ids', 'attention_mask', 'labels']
    convert_to_nd = True

    def __init__(self,
                 filename,
                 func,
                 initargs=None,
                 initializer=None,
                 use_cache=True,
                 cache_params=None):
        if cache_params is not None:
            cache_file = self.cache_file(filename, params=cache_params)
        else:
            cache_file = None
        if use_cache and cache_file is not None and os.path.exists(cache_file):
            logger.info(f"  Loading docs and features from : {cache_file} ...")
            cache = self.load_cache(cache_file)
            self.examples, self.features = cache['examples'], cache['features']
            logger.info("  Finished !")
        else:
            self.examples = self.create_examples(filename)
            self.features = self.convert_examples_to_features(
                self.examples,
                func=func,
                initargs=initargs,
                initializer=initializer,
                threads=1
            )
            if use_cache and cache_file is not None:
                logger.info(f"  Writing examples and features to : {cache_file} ...")
                self.write_cache(
                    {'examples': self.docs, 'features': self.features},
                    cache_file
                )
                logger.info("  Finished !")

    def cache_file(self, ori_file, params: Dict):
        """
        根据原始文件和参数字典生成dataset的哈希值
        :param ori_file:
        :param params:
        :return:
        """
        string = f"{ori_file}-{self.__class__.__name__}-" + "{"
        for k, v in params.items():
            line = f"{k}:{v},"
            string += line
        string += "}"
        ori_dir, filename = os.path.split(ori_file)
        cache_name = f"{self.md5(string)}"
        return os.path.join(ori_dir, self.cache_dir, cache_name)

    @staticmethod
    def md5(content: str):
        md5gen = hashlib.md5()
        md5gen.update(content.encode())
        return md5gen.hexdigest()

    @classmethod
    def load_cache(cls, filename):
        return pickle.load(gzip.open(filename, "rb"))
        # return torch.load(filename, map_location='cpu')

    @classmethod
    def write_cache(cls, data, filename):
        save_dir, _ = os.path.split(filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # torch.save(data, filename)
        with gzip.open(filename, "wb") as file:
            pickle.dump(data, file)

    def data(self):
        return self.examples

    def correct_offsets(self, predictions, indexes: Union[List[int], int]):
        """
        根据原本feature的offset，将预测的位置还原回去
        :param predictions:
        :param indexes: 需要纠正的值所在下标
        :return:
        """
        assert len(predictions) == len(
            self.features), f"predictions :{len(predictions)} != features : {len(self.features)}"
        if isinstance(indexes, int):
            indexes = [indexes]
        correct_res = []
        for prediction, feature in zip(predictions, self.features):
            span = set()
            for pred in prediction:
                temp = ()
                for i, p in enumerate(pred):
                    if i in indexes:
                        if feature.offsets[p] is not None:
                            temp += (feature.offsets[p],)
                    else:
                        temp += (p,)
                span.add(temp)
            correct_res.append(span)
        return correct_res

    @classmethod
    def create_examples(cls, filename) -> List[InputExample]:
        examples = []
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                entity_map = {}
                entities = []
                for entity_mention in d['entity_mentions']:
                    entity_map[entity_mention['id']] = entity_mention
                    entities.append(Span(
                        start=entity_mention['start'],
                        end=entity_mention['end'] - 1,
                        label=entity_mention['entity_type']
                    ))
                events = []
                for event_mention in d['event_mentions']:
                    trigger = event_mention['trigger']
                    arguments = []
                    for argument in event_mention['arguments']:
                        entity = entity_map[argument['entity_id']]
                        arguments.append(Span(
                            start=entity['start'],
                            end=entity['end'] - 1,
                            label=argument['role']
                        ))
                    event = Event(
                        trigger=Span(
                            start=trigger['start'],
                            end=trigger['end'] - 1,
                            label=event_mention['event_type']),
                        arguments=arguments
                    )
                    events.append(event)
                example = InputExample(
                    doc_id=d['sent_id'],
                    sent_id=d['sent_id'],
                    words=d['tokens'],
                    events=events,
                    entities=entities
                )
                examples.append(example)
        return examples

    @classmethod
    def convert_examples_to_features(
            cls,
            examples: List[InputExample],
            func,
            initargs=None,
            initializer=None,
            threads=1):
        threads = min(cpu_count(), threads)
        if threads <= 1:
            features = []
            if initializer is not None:
                initializer(*initargs)
            for ex_idx, example in enumerate(tqdm(examples, desc="processing")):
                feature = func(example)
                features.append(feature)
        else:
            features = []
            with Pool(threads, initializer=initializer, initargs=initargs) as p:
                features = list(tqdm(
                    p.imap(func, examples, chunksize=32),
                    total=len(examples),
                    desc="processing"
                ))
        new_features = []
        for feature in features:
            if isinstance(feature, list):
                new_features.extend(feature)
            else:
                new_features.append(feature)
        return new_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        res = {}
        for input_name in self.input_names:
            value = getattr(feature, input_name)
            if self.convert_to_nd:
                value = np.array(value)
            res[input_name] = value
        return res


def fill(data, new_data, axis=[0, 1]):
    for j, x in enumerate(data):
        if len(x.size()) > 2:
            new_data[j, :, :x.shape[axis[0]], :x.shape[axis[1]]] = x
        else:
            new_data[j, :x.shape[axis[0]], :x.shape[axis[1]]] = x
    return new_data


def create_mh_labels(spans, max_length, label2id, seq_len=None, task_type='sigmoid'):
    """
    创建multi head矩阵的labels，global pointer和biaffine需要
    :param task_type: 任务类型
    :param spans: (start, end, label_type)
    :param max_length: 最大长度
    :param label2id:
    :param seq_len: 当前句子的真实长度
    :return:
    """
    if task_type == 'sigmoid':
        labels = np.zeros((len(label2id), max_length, max_length), dtype=np.int)
    else:
        labels = np.zeros((max_length, max_length), dtype=np.int)
    for span in spans:
        label_id = label2id[span[2]]
        start = span[0]
        if seq_len is None:
            end = min(max_length - 1, span[1])
        else:
            end = min(seq_len - 1, span[1])
        if start <= end:
            if task_type == 'sigmoid':
                labels[label_id, start, end] = 1
            else:
                labels[start, end] = label_id
    return labels
