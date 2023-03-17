# -*- coding: UTF-8 -*-
# author    : huanghui
import torch
import torch.nn as nn
from .base import (
    get_word_feature, Biaffine, MLP)


class MgreeLayer(nn.Module):
    def __init__(self, hidden_size, num_labels, vocab_size):
        super(MgreeLayer, self).__init__()
        self.num_labels = num_labels
        self.glove = nn.Embedding(vocab_size, 100)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            hidden_size + 100,
            # hidden_size,
            hidden_size // 2, 1,
            batch_first=True,
            bidirectional=True)
        biaffine_size = 768
        self.start = MLP(hidden_size, biaffine_size, activation='gelu')
        self.end = MLP(hidden_size, biaffine_size, activation='gelu')
        self.biaffine = Biaffine(
            n_in=biaffine_size, n_out=num_labels
        )

    def forward(self, bert_outputs, pieces2word, word_ids, loss_mask=None):
        # sequence_output = torch.stack(bert_outputs.hidden_states[-3:], dim=-1).mean(-1)
        sequence_output = bert_outputs[0]
        sequence_output = self.dropout(sequence_output)

        glove_emb = self.glove(word_ids)  # bs, seq, 100
        word_feature = get_word_feature(sequence_output, pieces2word)
        concat_output = torch.cat([glove_emb, word_feature], dim=-1)
        lstm_output, _ = self.lstm(concat_output)

        start = self.start(word_feature)
        end = self.end(word_feature)

        scores = self.biaffine(start, end)
        if loss_mask is not None:
            scores = scores * loss_mask - (1 - loss_mask) * 1e12
        return scores, 0
