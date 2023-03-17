# -*- coding: UTF-8 -*-
# author    : huanghui
import logging
import os.path
import time
from abc import ABC
import numpy as np
import torch
from packaging import version
from .base import Trainer, BaseModel
from utils import load_from_json, set_seed, load_embeddings
from dataset.mgree import MgreeDataset, MgreeCollator
from dataset.base import InputExample, Result, Span, Event
from layers.loss import gp_loss
from layers.mgree import MgreeLayer

logger = logging.getLogger("main")


class MgreeModel(BaseModel):
    def __init__(self, model_name_or_path, num_tri, num_arg, from_scratch=False, vocab_size=1, only_trigger=False):
        super(MgreeModel, self).__init__(model_name_or_path, from_scratch)
        self.only_trigger = only_trigger
        self.num_tri = num_tri
        self.num_arg = num_arg
        self.num_labels = self.num_tri + self.num_arg + 1
        self.cls = MgreeLayer(self.hidden_size, self.num_labels, vocab_size)

    def forward(self,
                input_ids,
                attention_mask,
                pieces2word,
                word_mask,
                word_ids,
                loss_mask=None,
                labels=None,
                inputs_embeds=None):
        bert_outputs = self.get_embeddings(input_ids, attention_mask, inputs_embeds=inputs_embeds)
        scores, _ = self.cls(bert_outputs, pieces2word, word_ids, loss_mask)
        outputs = (scores,)
        # outputs = (torch.sigmoid(scores), )
        if labels is not None:
            loss = gp_loss(scores, labels, self.num_labels)
            outputs = (loss,) + outputs
        return outputs


class MgreeTrainer(Trainer, ABC):
    label_names = ["labels"]
    trigger_use_bio = False
    trigger_default_none = None  # 增加一个O标签
    argument_use_bio = False
    argument_default_none = None
    logit_indexs = [1]
    # thresholds = {
    #     "trigger": 0,
    #     "argument": 0,
    #     "relation": 0
    # }
    thresholds = {
        "trigger": 0.1,
        "argument": -1.2,
        "relation": -0.2
    }
    # thresholds = {
    #     "trigger": 0.5,
    #     "argument": 0.5,
    #     "relation": 0.5
    # }
    only_trigger = False  # 只做触发词抽取

    def __init__(self, args, from_scratch):
        super(MgreeTrainer, self).__init__(args, from_scratch)

        self.event_role = load_from_json(os.path.join(self.data_dir, self.event_role_file))

        self.vocab2id, pretrained_embed = load_embeddings("data/100.utf8")
        set_seed(self.args.seed)
        self.model = MgreeModel(self.model_dir, len(self.trigger2id),
                                len(self.argument2id), from_scratch,
                                vocab_size=len(self.vocab2id), only_trigger=self.only_trigger)
        self.model.cls.glove.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.model.init_weights()
        self.collate_fn = MgreeCollator(self.args.single_word)
        self.set_device()
        if self.args.compile and version.parse(torch.__version__) >= version.parse("1.10"):
            self.model = torch.compile(self.model)

    def create_dataset(self, filename, training=False):
        dataset = MgreeDataset(
            filename,
            tokenizer=self.tokenizer,
            trigger2id=self.trigger2id,
            argument2id=self.argument2id,
            vocab2id=self.vocab2id,
            max_length=self.args.max_length
        )
        return dataset

    def decode_tri(self, prediction, mask):
        """
        :param prediction: [label_num, seq_len, seq_len]
        :param mask:
        :return:
        """
        trigger_set = set()
        seq_len = sum(mask)
        labels, heads, tails = np.where(prediction > self.thresholds['trigger'])
        for i, (label_id, head_id, tail_id) in enumerate(zip(labels, heads, tails)):
            if head_id >= seq_len or tail_id >= seq_len:
                continue
            if self.args.single_word and head_id != tail_id:
                continue
            label = self.id2trigger[label_id]
            if head_id <= tail_id:
                trigger_set.add((head_id, tail_id, label))
        tri_map = {tri[0]: tri for tri in trigger_set}
        return trigger_set, tri_map

    def decode_arg(self, prediction, mask):
        argument_set = set()
        seq_len = sum(mask)
        heads, tails = np.where(prediction > self.thresholds['argument'])
        for i, (head_id, tail_id) in enumerate(zip(heads, tails)):
            if head_id >= seq_len or tail_id >= seq_len:
                continue
            if head_id <= tail_id:
                argument_set.add((head_id, tail_id))
        arg_map = {arg[0]: arg for arg in argument_set}
        return argument_set, arg_map

    def decode_rel(self, prediction, mask, tri_map, arg_map):
        """
        :param prediction: [arg_num, seq_len, seq_len]
        :param mask: []
        :param tri_map
        :param arg_map
        :return:
        """
        seq_len = sum(mask)
        event_set = {}
        labels, heads, tails = np.where(prediction > self.thresholds['relation'])

        for i, (label_idx, arg_idx, tri_idx) in enumerate(zip(labels, heads, tails)):
            if arg_idx >= seq_len or tri_idx >= seq_len:
                continue
            label = self.id2argument[label_idx]
            if arg_idx in arg_map and tri_idx in tri_map and label in self.event_role[tri_map[tri_idx][2]]:
                arg_mention = arg_map[arg_idx]
                argument = arg_mention + (label,)
                if tri_idx not in event_set:
                    event_set[tri_idx] = set()
                event_set[tri_idx].add(argument)
        return event_set

    def decode_batch(self, tri_predictions, arg_predictions, rel_predictions, masks):
        batch_results = []
        for tri_pred, arg_pred, rel_pred, mask in zip(tri_predictions, arg_predictions, rel_predictions, masks):
            triggers, tri_map = self.decode_tri(tri_pred, mask)
            arg_mentions, arg_map = self.decode_arg(arg_pred, mask)
            event_set = self.decode_rel(rel_pred, mask, tri_map, arg_map)
            events = []
            for trigger in triggers:
                arguments = []
                if trigger[0] in event_set:
                    for arg in event_set[trigger[0]]:
                        arguments.append(Span(arg[0], arg[1], arg[2]))
                events.append(Event(trigger=Span(*trigger), arguments=arguments))
            result = Result(
                events=events
            )
            batch_results.append(result)
        return batch_results

    def decode(self, scores, masks):
        all_results = []
        for score, mask in zip(scores, masks):
            tri_score = score[:, :len(self.trigger2id)]
            if not self.only_trigger:
                arg_score = score[:, len(self.trigger2id)]
                rel_score = score[:, -len(self.argument2id):]
            else:
                bsz = score.shape[0]
                seq_len = score.shape[2]
                arg_score = np.zeros((bsz, seq_len, seq_len))
                rel_score = np.zeros((bsz, len(self.argument2id), seq_len, seq_len))
            results = self.decode_batch(tri_score, arg_score, rel_score, mask)
            all_results.extend(results)
        return all_results

    def _predict(self, dataset):
        predict_dataloader = self.create_dataloader(dataset, training=False)
        scores, masks = [], []
        for batch in predict_dataloader:
            # score = self.detach_tensor(batch['labels'])
            outputs = self.predict_step(batch)[0]
            score = self.detach_tensor(outputs)
            mask = self.detach_tensor(batch['word_mask'])
            scores.append(score)
            masks.append(mask)
        return scores, masks

    def predict(self, dataset):
        t1 = time.time()
        scores, masks = self._predict(dataset)
        results = self.decode(scores, masks)
        golds = dataset.data()
        for i in range(len(golds)):
            results[i].doc_id = golds[i].doc_id
            results[i].sent_id = golds[i].sent_id
        t2 = time.time()
        print(f"total: {t2-t1}s; avg: {(t2 - t1)/len(dataset)}")
        return golds, results
