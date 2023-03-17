# -*- coding: UTF-8 -*-
# author    : huanghui
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import get_activation


def sequence_mask(x, mask, mask_tril=True, mask_triu=False):
    # padding mask
    pad_mask = mask.unsqueeze(1).unsqueeze(1).expand_as(x)
    x = x * pad_mask - (1 - pad_mask) * 1e12
    # 排除下三角
    if mask_tril:
        tril_mask = torch.tril(torch.ones_like(x), -1)
        x = x - tril_mask * 1e12
    if mask_triu:
        triu_mask = torch.triu(torch.ones_like(x), 1)
        x = x - triu_mask * 1e12
    return x


def get_word_feature_mean(sequence_output, pieces2word):
    """

    :param sequence_output: [bs, tok_len, h]
    :param pieces2word: [bs, word, tok]
    :return:
    """
    pieces2word = pieces2word.long()
    length_tensor = pieces2word.sum(dim=2).unsqueeze(2)
    sum_vector = torch.bmm(pieces2word.float(), sequence_output)
    avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
    return avg_vector


def get_word_feature_max(sequence_output, pieces2word):
    """

    :param sequence_output: [bs, tok_len, h]
    :param pieces2word: [bs, word, tok]
    :return:
    """
    length = pieces2word.size(1)
    min_value = torch.min(sequence_output).item()
    # Max pooling word representations from pieces
    _bert_embs = sequence_output.unsqueeze(1).expand(-1, length, -1, -1)
    _bert_embs = torch.masked_fill(_bert_embs, pieces2word.eq(0).unsqueeze(-1), min_value)
    word_reps, _ = torch.max(_bert_embs, dim=2)
    return word_reps


def get_word_feature(sequence_output, pieces2word, mode='max'):
    if mode == 'max':
        return get_word_feature_max(sequence_output, pieces2word)
    return get_word_feature_mean(sequence_output, pieces2word)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self,
                 hidden_size,
                 max_position_embeddings=512):
        super(SinusoidalPositionEmbedding, self).__init__()
        position_ids = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(-1)
        indices = torch.arange(0, hidden_size // 2, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / hidden_size)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        self.embeddings = torch.reshape(embeddings, (-1, max_position_embeddings, hidden_size))

    def forward(self, inputs=None):
        pos = self.embeddings[:, :inputs.size()[1]].to(inputs.device)  # 第二个维度为长度
        if len(inputs.size()) > 3:
            cos_pos = pos[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., None, ::2].repeat_interleave(2, dim=-1)
        else:
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
        inputs2 = torch.stack([-inputs[..., 1::2], inputs[..., ::2]], dim=-1)
        inputs2 = inputs2.reshape(inputs.shape)
        inputs = inputs * cos_pos + inputs2 * sin_pos
        return inputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)  # b,x,1 + i
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)  # b,y,1 + i
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s


class MLP(nn.Module):
    def __init__(self, in_size, out_size, activation=None, dropout=0):
        super(MLP, self).__init__()
        self.dense = nn.Linear(in_size, out_size)
        self.activation = activation.lower() if activation is not None else "linear"
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = get_activation(self.activation)(self.dense(inputs))
        return self.dropout(outputs)
