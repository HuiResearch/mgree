# -*- coding: UTF-8 -*-
# author    : huanghui
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


def multi_label_categorical_ce(inputs: torch.Tensor, targets: torch.Tensor):
    y_pred = (1 - 2 * targets) * inputs  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - targets * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - targets) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def sparse_multi_label_ce(y_pred: torch.Tensor, y_true: torch.Tensor, mask_zero=False):
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], axis=-1)
    y_pos_2 = y_pred.gather(dim=-1, index=y_true)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = y_pred.gather(dim=-1, index=y_true)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-8, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    return pos_loss + neg_loss


def sequence_loss(logits, labels, mask, num_labels):
    loss_fct = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss


def gp_loss(logits, labels, heads):
    batch_size = logits.shape[0]
    y_true = labels.reshape(batch_size * heads, -1)
    y_pred = logits.reshape(batch_size * heads, -1)
    loss = multi_label_categorical_ce(y_pred, y_true)
    return loss


def sparse_loss(logits, labels, heads, mask_zero=True):
    batch_size = logits.shape[0]
    y_true = labels.reshape(batch_size * heads, -1)
    y_pred = logits.reshape(batch_size * heads, -1)
    loss = sparse_multi_label_ce(y_pred, y_true, mask_zero)
    return loss.mean()


def focal_loss(logits, labels, heads):
    loss_fct = BinaryFocalLoss(alpha=2, gamma=0.5, ignore_index=None)
    batch_size = logits.shape[0]
    y_true = labels.reshape(batch_size * heads, -1)
    y_pred = logits.reshape(batch_size * heads, -1)
    loss = loss_fct(y_pred, y_true)
    return loss
