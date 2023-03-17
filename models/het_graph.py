import logging
import os.path
from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from .base import Trainer, BaseModel
from utils import load_from_json, set_seed, load_embeddings
from dataset.het_graph import HetGraphDataset, HetGraphCollator
from dataset.base import Result, Span, Event
from layers.mgree import MgreeLayer

logger = logging.getLogger("main")


class HetGraphModel(BaseModel):
    def __init__(self, model_name_or_path, num_labels, from_scratch=False, vocab_size=1):
        super(HetGraphModel, self).__init__(model_name_or_path, from_scratch)
        self.num_labels = num_labels
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
        scores, _ = self.cls(bert_outputs, pieces2word, word_ids, None)
        scores = scores.permute(0, 2, 3, 1)
        outputs = (scores,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores[loss_mask.bool()], labels[loss_mask.bool()])
            outputs = (loss,) + outputs
        return outputs


class HetGraphTrainer(Trainer, ABC):
    label_names = ["labels"]
    trigger_use_bio = False
    trigger_default_none = None  # 增加一个O标签
    argument_use_bio = False
    argument_default_none = None
    logit_indexs = [1]

    def __init__(self, args, from_scratch):
        super(HetGraphTrainer, self).__init__(args, from_scratch)

        self.event_role = load_from_json(os.path.join(self.data_dir, self.event_role_file))

        labels = ['NA', "argument"]
        for t, _ in self.trigger2id.items():
            labels.append(t)
        for t, _ in self.argument2id.items():
            labels.append(t)
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        print(self.label2id)
        self.vocab2id, pretrained_embed = load_embeddings("data/100.utf8")
        set_seed(self.args.seed)
        self.model = HetGraphModel(
            self.model_dir, len(labels), from_scratch,
            vocab_size=len(self.vocab2id))
        self.model.cls.glove.weight.data.copy_(torch.from_numpy(pretrained_embed))

        self.model.init_weights()
        self.collate_fn = HetGraphCollator()
        self.set_device()

    def create_dataset(self, filename, training=False):
        dataset = HetGraphDataset(
            filename,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            vocab2id=self.vocab2id,
            max_length=self.args.max_length
        )
        return dataset

    def decode_single(self, prediction, mask):
        seq_len = sum(mask)
        labels = np.argmax(prediction, axis=-1)  # seq, seq
        triggers, arguments = set(), set()
        role_map = {}
        for start, end in zip(*np.where(labels > 0)):
            label_id = labels[start, end]
            label = self.id2label[label_id]
            if label == "argument":
                if 0 <= start <= end < seq_len:
                    arguments.add((start, end))
            elif label in self.trigger2id:
                if 0 <= start <= end < seq_len:
                    if ("ace05" in self.args.data_dir and "+" not in self.args.data_dir) and start != end:
                        continue
                    triggers.add((start, end, label))
            elif label in self.argument2id:
                if 0 <= start < seq_len and 0 <= end < seq_len:
                    if end not in role_map:
                        role_map[end] = set()
                    role_map[end].add((start, label))  # tri_idx: arg_idx, role_type
        argument_map = {arg[0]: arg for arg in arguments}
        events = []
        for trigger in triggers:
            cur_arguments = set()
            if trigger[0] in role_map:
                for mapping in role_map[trigger[0]]:
                    arg_head_idx, role_type = mapping
                    if arg_head_idx in argument_map and role_type in self.event_role[trigger[2]]:
                        cur_arguments.add((arg_head_idx, argument_map[arg_head_idx][1], role_type))
            events.append(Event(trigger=Span(*trigger), arguments=[Span(*arg) for arg in cur_arguments]))
        return events

    def decode_batch(self, predictions, masks):
        batch_results = []
        for prediction, mask in zip(predictions, masks):
            events = self.decode_single(prediction, mask)
            result = Result(
                events=events
            )
            batch_results.append(result)
        return batch_results

    def decode(self, predictions, masks):
        all_results = []
        for prediction, mask in zip(predictions, masks):
            results = self.decode_batch(prediction, mask)
            all_results.extend(results)
        return all_results

    def _predict(self, dataset):
        predict_dataloader = self.create_dataloader(dataset, training=False)
        scores, masks = [], []
        for batch in predict_dataloader:
            outputs = self.predict_step(batch)[0]
            score = self.detach_tensor(outputs)
            mask = self.detach_tensor(batch['word_mask'])
            scores.append(score)
            masks.append(mask)
        return scores, masks

    def predict(self, dataset):
        scores, masks = self._predict(dataset)
        results = self.decode(scores, masks)
        golds = dataset.data()
        for i in range(len(golds)):
            results[i].doc_id = golds[i].doc_id
            results[i].sent_id = golds[i].sent_id
        return golds, results
