# -*- coding: UTF-8 -*-
# author    : huanghui
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .base import BaseDataset, InputExample, InputFeature, fill
from transformers import PreTrainedTokenizerBase
from abc import ABC
from functools import partial


def convert_example_to_feature_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_example_to_feature(
        example: InputExample, label2id, vocab2id, max_length):
    input_word_ids = []
    for word in example.words:
        if word in vocab2id:
            input_word_ids.append(vocab2id[word])
        else:
            input_word_ids.append(vocab2id['<UNK>'])
    encoded_output = tokenizer(
        example.words, add_special_tokens=True, padding=False,
        truncation=True, max_length=max_length, is_split_into_words=True)
    input_ids = encoded_output['input_ids']
    word_ids = encoded_output.word_ids()
    length = max([i for i in word_ids if i is not None]) + 1
    input_word_ids = input_word_ids[:length]

    labels = np.zeros((length, length))
    pieces2word = np.zeros((length, len(input_ids)), dtype=bool)

    for tok_id, word_id in enumerate(encoded_output.word_ids()):
        if word_id is None:
            continue
        # if word_id in offset_map:
        pieces2word[word_id, tok_id] = 1

    for event in example.events:
        t_s = event.trigger.start
        t_e = event.trigger.end
        label = event.trigger.label
        if t_s <= t_e < length:
            labels[t_s, t_e] = label2id[label]

        for argument in event.arguments:
            a_s = argument.start
            a_e = argument.end
            arg_label = argument.label
            if a_s <= a_e < length:
                labels[a_s, a_e] = label2id["argument"]
            if a_s < length and t_s < length:
                labels[a_s, t_s] = label2id[arg_label]

    return InputFeature(
        guid=example.doc_id + example.sent_id,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        word_ids=input_word_ids,
        labels=[labels],
        pieces2word=pieces2word,
        offsets=word_ids
    )


class HetGraphDataset(BaseDataset, ABC):
    input_names = ['input_ids', 'attention_mask', 'word_ids', 'labels', 'pieces2word']
    convert_to_nd = False

    def __init__(
            self,
            filename, tokenizer,
            label2id, vocab2id, max_length):
        super(HetGraphDataset, self).__init__(
            filename,
            func=partial(convert_example_to_feature,
                         label2id=label2id,
                         vocab2id=vocab2id,
                         max_length=max_length),
            initargs=(tokenizer,),
            initializer=convert_example_to_feature_init,
            use_cache=False
        )


class HetGraphCollator:

    def __call__(self, data):
        input_ids = []
        attention_mask = []
        labels = []
        word_ids = []
        pieces2word = []
        word_mask = []
        max_tok = 0
        max_word = 0
        for d in data:
            input_ids.append(torch.LongTensor(d['input_ids']))
            attention_mask.append(torch.LongTensor(d['attention_mask']))
            word_ids.append(torch.LongTensor(d['word_ids']))
            labels.append(torch.LongTensor(d['labels'][0]))
            pieces2word.append(torch.LongTensor(d['pieces2word']))
            length = d['pieces2word'].shape[0]
            word_mask.append(torch.LongTensor([1] * length))
            if len(d['input_ids']) > max_tok:
                max_tok = len(d['input_ids'])
            if length > max_word:
                max_word = length

        input_ids = pad_sequence(input_ids, True)
        attention_mask = pad_sequence(attention_mask, True)
        word_ids = pad_sequence(word_ids, True)
        word_mask = pad_sequence(word_mask, True)

        bsz = len(data)
        labels_matrix = torch.zeros((bsz, max_word, max_word), dtype=torch.long)
        labels_matrix = fill(labels, labels_matrix, axis=[0, 1])

        pieces2word_matrix = torch.zeros((len(data), max_word, max_tok), dtype=torch.bool)
        pieces2word_matrix = fill(pieces2word, pieces2word_matrix, axis=[0, 1])

        grid_mask = word_mask.unsqueeze(1)
        grid_mask = grid_mask.expand([bsz, max_word, max_word]).clone()

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_ids": word_ids,
                "labels": labels_matrix,
                "pieces2word": pieces2word_matrix,
                "loss_mask": grid_mask,
                "word_mask": word_mask}
