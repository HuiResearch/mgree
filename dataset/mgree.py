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
        example: InputExample, trigger2id, argument2id,
        vocab2id, max_length):
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

    triggers = np.zeros((len(trigger2id), length, length))
    arguments = np.zeros((1, length, length))
    relations = np.zeros((len(argument2id), length, length))
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
            triggers[trigger2id[label], t_s, t_e] = 1

        for argument in event.arguments:
            a_s = argument.start
            a_e = argument.end
            arg_label = argument.label
            if a_s <= a_e < length:
                arguments[0, a_s, a_e] = 1
            if a_s < length and t_s < length:
                relations[argument2id[arg_label], a_s, t_s] = 1

    return InputFeature(
        guid=example.doc_id + example.sent_id,
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        word_ids=input_word_ids,
        labels=[triggers, arguments, relations],
        pieces2word=pieces2word,
        offsets=word_ids
    )


class MgreeDataset(BaseDataset, ABC):
    input_names = ['input_ids', 'attention_mask', 'word_ids', 'labels', 'pieces2word']
    convert_to_nd = False

    def __init__(
            self, filename, tokenizer,
            trigger2id, argument2id,
            vocab2id, max_length):
        super(MgreeDataset, self).__init__(
            filename,
            func=partial(convert_example_to_feature,
                         trigger2id=trigger2id,
                         argument2id=argument2id,
                         vocab2id=vocab2id,
                         max_length=max_length),
            initargs=(tokenizer,),
            initializer=convert_example_to_feature_init,
            use_cache=False
        )


class MgreeCollator:
    def __init__(self, single_word):
        self.single_word = single_word

    def __call__(self, data):
        input_ids = []
        attention_mask = []
        triggers = []
        arguments = []
        relations = []
        word_ids = []
        pieces2word = []
        word_mask = []
        max_tok = 0
        max_word = 0
        for d in data:
            input_ids.append(torch.LongTensor(d['input_ids']))
            attention_mask.append(torch.LongTensor(d['attention_mask']))
            word_ids.append(torch.LongTensor(d['word_ids']))
            triggers.append(torch.LongTensor(d['labels'][0]))
            relations.append(torch.LongTensor(d['labels'][2]))
            arguments.append(torch.LongTensor(d['labels'][1]))
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

        trigger_matrix = torch.zeros((len(data), triggers[0].shape[0], max_word, max_word), dtype=torch.long)
        trigger_matrix = fill(triggers, trigger_matrix, axis=[1, 2])

        argument_matrix = torch.zeros((len(data), 1, max_word, max_word), dtype=torch.long)
        argument_matrix = fill(arguments, argument_matrix, axis=[1, 2])

        relation_matrix = torch.zeros((len(data), relations[0].shape[0], max_word, max_word), dtype=torch.long)
        relation_matrix = fill(relations, relation_matrix, axis=[1, 2])

        pieces2word_matrix = torch.zeros((len(data), max_word, max_tok), dtype=torch.bool)
        pieces2word_matrix = fill(pieces2word, pieces2word_matrix, axis=[0, 1])

        labels = torch.cat([trigger_matrix, argument_matrix, relation_matrix], dim=1)

        loss_mask = word_mask.unsqueeze(1).unsqueeze(1)
        rel_mask = loss_mask.expand([len(data), relations[0].shape[0], max_word, max_word])
        tri_mask = loss_mask.expand([len(data), triggers[0].shape[0], max_word, max_word])

        arg_mask = loss_mask.expand([len(data), 1, max_word, max_word])
        if self.single_word:
            triu_mask = torch.triu(tri_mask)
            tril_mask = torch.tril(tri_mask)
            tri_mask = triu_mask & tril_mask
            arg_mask = torch.triu(arg_mask)
            tri_mask = torch.cat([tri_mask, arg_mask], dim=1)
        else:
            tri_mask = torch.triu(torch.cat([tri_mask, arg_mask], dim=1))

        loss_mask = torch.cat([tri_mask, rel_mask], dim=1)

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "word_ids": word_ids,
                "labels": labels,
                "pieces2word": pieces2word_matrix,
                "loss_mask": loss_mask,
                "word_mask": word_mask}
