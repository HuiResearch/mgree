# -*- coding: UTF-8 -*-
# author    : huanghui
# date      : 2021/8/10 19:23
# file name : training_utils.py
# project   : ydlj
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from collections import defaultdict
from torch import nn
from typing import List, Union, Tuple


def _model_unwrap(model: nn.Module) -> nn.Module:
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return _model_unwrap(model.module)
    else:
        return model


class FGM:
    def __init__(
            self,
            model,
            epsilon=1.,
            param_name='word_embeddings'):
        self.model = _model_unwrap(model)
        self.epsilon = epsilon
        self.param_name = param_name
        self.backup = {}

    def attack(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model, adv_steps=3, epsilon=1., alpha=0.3, param_name='word_embeddings'):
        self.model = _model_unwrap(model)
        self.emb_backup = {}
        self.grad_backup = {}
        self.adv_steps = adv_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.param_name = param_name

    def attack(self, is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class AverageModel:
    def __init__(self, model):
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.registered = False
        self.updated = False

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.registered = True

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class SWA(AverageModel):
    def __init__(self, model):
        super(SWA, self).__init__(model)
        self.n_averaged = 0

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.shadow[name] + (param.data - self.shadow[name]) / (self.n_averaged + 1)
                self.shadow[name] = new_average.clone()
        self.n_averaged += 1
        self.updated = True


class EMA(AverageModel):
    def __init__(self, model, decay: Union[float, Tuple[float, float]] = (0.999, 0.999)):
        super(EMA, self).__init__(model)
        if isinstance(decay, float):
            decay = (decay, decay)
        else:
            assert len(decay) == 2 and all([0 <= d <= 1 for d in decay])
        self.decay = decay
        self.changed = False  # 标识使用哪一个decay

    def update(self):
        decay = self.decay[0] if not self.changed else self.decay[1]
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
        self.updated = True


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)
